#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate a raw .bin file for the *current* TensorFlow MiniCRN_Causal128 model.

Usage example:
python gen_minicrn_tf_input_bin.py --out ./minicrn_tf_input.bin --mode white --seed 1234

Output:
  float32 little-endian
  shape = [1, 51, 200, 1]   (channels_last)

This matches:
  train_minicrn_causal128_tbptt_tf2_batch4.py
"""

import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path


# ===================== Model-aligned constants =====================
SAMPLE_RATE = 16000
N_FFT = 100
HOP_LENGTH = 25
CHUNK_FRAMES = 200
FREQ_BINS = N_FFT // 2 + 1   # 51


# ===================== Signal generation =====================
def generate_waveform(n_samples, sr, seed, mode):
    rng = np.random.default_rng(seed)

    if mode == "white":
        x = rng.standard_normal(n_samples, dtype=np.float32)

    elif mode == "colored":
        w = rng.standard_normal(n_samples, dtype=np.float32)
        a = 0.98
        y = np.zeros_like(w)
        acc = np.float32(0.0)
        for i in range(len(w)):
            acc = a * acc + (1.0 - a) * w[i]
            y[i] = acc
        x = y

    elif mode == "tones":
        t = np.arange(n_samples, dtype=np.float32) / sr
        x = np.zeros_like(t)
        for _ in range(rng.integers(2, 5)):
            f = rng.uniform(200.0, 3800.0)
            a = rng.uniform(0.2, 0.6)
            p = rng.uniform(0.0, 2 * np.pi)
            x += a * np.sin(2 * np.pi * f * t + p)
        x += 0.05 * rng.standard_normal(n_samples, dtype=np.float32)

    else:
        raise ValueError("Unknown mode")

    x /= max(np.max(np.abs(x)), 1e-8)
    return x.astype(np.float32)


# ===================== STFT (TF-consistent) =====================
def stft_mag_tf(wav):
    """
    wav: [1, S] float32
    return: [1, F, T, 1]
    """
    stft = tf.signal.stft(
        wav,
        frame_length=N_FFT,
        frame_step=HOP_LENGTH,
        fft_length=N_FFT,
        window_fn=tf.signal.hann_window,
        pad_end=False,
    )                       # [1, T, F]
    mag = tf.abs(stft)
    mag = tf.transpose(mag, [0, 2, 1])   # [1, F, T]
    return mag[..., tf.newaxis]          # [1, F, T, 1]


# ===================== Main =====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output .bin file")
    ap.add_argument("--mode", default="white", choices=["white", "colored", "tones"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # waveform length needed for exactly T frames
    n_samples = (CHUNK_FRAMES - 1) * HOP_LENGTH + N_FFT

    wav = generate_waveform(n_samples, SAMPLE_RATE, args.seed, args.mode)
    wav = wav[None, :]   # [1, S]

    mag = stft_mag_tf(tf.convert_to_tensor(wav)).numpy()

    assert mag.shape == (1, FREQ_BINS, CHUNK_FRAMES, 1), mag.shape

    out = mag.astype("<f4")  # float32 little-endian

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.tofile(args.out)

    print("OK")
    print(f"output file : {args.out}")
    print(f"shape       : {out.shape}")
    print(f"dtype       : float32 (little-endian)")


if __name__ == "__main__":
    main()
