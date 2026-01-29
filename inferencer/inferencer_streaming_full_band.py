# inferencer/inferencer_streaming_full_band.py
#
# Streaming full-band inference with a FIXED 4-frame receptive window:
#   - Each inference step sees exactly 4 frames: [t-3, t-2, t-1, t]
#   - Only the LAST frame (t) output magnitude is used to reconstruct audio
#   - LSTM state is carried across steps if the model supports it
#
# Designed to work well with MiniCRN_Causal128 AFTER you add a stateful forward:
#   out, state = model(x, state)    where x is [B,1,F,T] and state is (h,c)
#
# If the model does not support state passing/returning, this code still runs
# but quality may degrade because LSTM resets at every step.

import numpy as np
import torch
import librosa
from collections import deque


@torch.no_grad()
def streaming_full_band(model, device, inference_args, noisy, sr=24000):
    """
    Args:
        model: enhancement model (e.g., MiniCRN_Causal128)
        device: torch device
        inference_args: dict with keys
            - n_fft, hop_length, win_length
            - target_sr (optional, default=16000)
            - lookback_frames (optional, default=4; MUST be 4 for your constraint)
        noisy: 1-D numpy array waveform
        sr: sampling rate of 'noisy'
    Returns:
        enhanced (numpy 1-D), sr_out
    """
    # ------------------------
    # Config
    # ------------------------
    n_fft = int(inference_args["n_fft"])
    hop = int(inference_args["hop_length"])
    win = int(inference_args["win_length"])
    target_sr = int(inference_args.get("target_sr", 16000))
    lookback = int(inference_args.get("lookback_frames", 4))
    if lookback != 4:
        # You asked specifically for 4 frames: 1 current + 3 past
        lookback = 4

    # ------------------------
    # Resample to model SR
    # ------------------------
    orig_sr = int(sr)
    orig_len = int(len(noisy))

    if sr != target_sr:
        noisy_rs = librosa.resample(noisy, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    else:
        noisy_rs = noisy

    # Torch waveform
    x = torch.tensor(noisy_rs, device=device, dtype=torch.float32)

    # Ensure at least one analysis frame
    if x.numel() < win:
        x = torch.nn.functional.pad(x, (0, win - x.numel()))

    # Analysis/synthesis window
    window = torch.hann_window(win, device=device, dtype=torch.float32)

    # center=False style framing:
    # frame i uses samples [i*hop : i*hop+win]
    n_frames = 1 + (x.numel() - win) // hop
    leftover = (x.numel() - win) % hop
    if leftover != 0:
        pad = hop - leftover
        x = torch.nn.functional.pad(x, (0, pad))
        n_frames = 1 + (x.numel() - win) // hop

    # Output buffers for OLA
    y = torch.zeros_like(x)
    wsum = torch.zeros_like(x)

    # Rolling mag frame buffer: each item is [F]
    mag_buf = deque(maxlen=lookback)
    # We also need phase for current frame reconstruction: [F] complex (unit phasor)
    # Past phases are not required because we only synthesize current frame (t)
    # using out[:, :, :, -1] and phase(t).

    # ------------------------
    # Stateful model adapter
    # ------------------------
    model.eval()
    state = None

    def _detach_state(st):
        if st is None:
            return None
        if isinstance(st, (tuple, list)) and len(st) == 2:
            return (st[0].detach(), st[1].detach())
        return st

    def _model_forward(mag_4d, st):
        """
        Returns (out_4d, new_state).
        Supports:
          - model(mag_4d, st) -> (out, st) or out
          - model(mag_4d)     -> (out, st) or out
        """
        # Prefer explicit stateful call if possible
        if st is not None:
            try:
                out = model(mag_4d, st)
                if isinstance(out, (tuple, list)) and len(out) == 2:
                    return out[0], out[1]
                return out, st
            except TypeError:
                pass

        # Try stateful call even if st is None (some models accept it)
        try:
            out = model(mag_4d, st)
            if isinstance(out, (tuple, list)) and len(out) == 2:
                return out[0], out[1]
            return out, st
        except TypeError:
            pass

        # Stateless fallback
        out = model(mag_4d)
        if isinstance(out, (tuple, list)) and len(out) == 2:
            return out[0], out[1]
        return out, None

    # ------------------------
    # Streaming loop: 1 frame per step, but model sees 4 frames (rolling window)
    # ------------------------
    for i in range(n_frames):
        t0 = i * hop
        frame = x[t0:t0 + win] * window

        # rFFT: [F] complex
        X = torch.fft.rfft(frame, n=n_fft)
        mag = torch.abs(X)                      # [F]
        ph = X / (mag + 1e-12)                  # [F] complex, unit magnitude

        # Update rolling buffer
        mag_buf.append(mag)

        # Warm-up: pad with zeros until we have 4 frames
        while len(mag_buf) < lookback:
            mag_buf.appendleft(torch.zeros_like(mag))

        # Build model input [1,1,F,4] with ordering [t-3, t-2, t-1, t]
        mag_ft = torch.stack(list(mag_buf), dim=1)     # [F, 4]
        mag_4d = mag_ft.unsqueeze(0).unsqueeze(0)      # [1,1,F,4]

        # Run model
        out_4d, state = _model_forward(mag_4d, state)
        state = _detach_state(state)

        # Take only current frame output magnitude: [:, :, :, -1]
        out_mag_t = out_4d[0, 0, :, -1]                # [F]

        # Reconstruct current frame in time-domain and OLA
        Y = out_mag_t.to(torch.complex64) * ph         # [F] complex
        y_frame = torch.fft.irfft(Y, n=n_fft)[:win]    # [win]
        y_frame = y_frame * window                     # synthesis window

        y[t0:t0 + win] += y_frame
        wsum[t0:t0 + win] += window * window

    # Normalize OLA
    y = y / (wsum + 1e-12)

    # Trim back to resampled length
    y = y[: len(noisy_rs)]
    enhanced = y.detach().cpu().numpy()

    # Resample back to original SR if needed
    if orig_sr != sr:
        enhanced = librosa.resample(enhanced, orig_sr=sr, target_sr=orig_sr)
        sr = orig_sr

    # Best-effort length match
    if len(enhanced) != orig_len:
        if len(enhanced) > orig_len:
            enhanced = enhanced[:orig_len]
        else:
            enhanced = np.pad(enhanced, (0, orig_len - len(enhanced)))

    return enhanced, sr
