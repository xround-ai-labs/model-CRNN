import librosa
import torch
import numpy as np


def full_band_no_truncation(model, device, inference_args, noisy, sr=24000):
    """
    extract full_band spectra for inference, without truncation.
    若輸入取樣率非 16 kHz，會自動重採樣以符合模型輸入需求。
    """
    n_fft = inference_args["n_fft"]
    hop_length = inference_args["hop_length"]
    win_length = inference_args["win_length"]
    target_sr = 16000

    # === 若取樣率不是 16 kHz，自動重採樣 ===
    if sr != target_sr:
        orig_sr = sr
        noisy = librosa.resample(noisy, orig_sr=orig_sr, target_sr=target_sr)
        sr = target_sr
        print(f"🔄 Resampled from {orig_sr} Hz → {target_sr} Hz")

    # === STFT ===
    noisy_stft = librosa.stft(noisy, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    noisy_mag, noisy_phase = librosa.magphase(noisy_stft)

    # === 模型推論 ===
    noisy_mag_tensor = torch.tensor(noisy_mag, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [F,T]→[1,1,F,T]
    #with torch.no_grad():
        #enhanced_mag_tensor = model(noisy_mag_tensor)
    enhanced_mag_tensor = noisy_mag_tensor  # 暫時不經過模型，直接輸出原始頻譜以測試流程
    enhanced_mag = enhanced_mag_tensor.squeeze(0).squeeze(0).detach().cpu().numpy()  # [1,1,F,T]→[F,T]

    # === 對齊頻譜時間長度 ===
    min_T = min(enhanced_mag.shape[1], noisy_phase.shape[1])
    if enhanced_mag.shape[1] != noisy_phase.shape[1]:
        print(f"⚠️ Length mismatch: enhanced={enhanced_mag.shape[1]} vs phase={noisy_phase.shape[1]}, trimming to {min_T}")
    enhanced_mag = enhanced_mag[:, :min_T]
    noisy_phase = noisy_phase[:, :min_T]

    # === ISTFT ===
    enhanced = librosa.istft(enhanced_mag * noisy_phase,
                             hop_length=hop_length,
                             win_length=win_length,
                             length=len(noisy))

    # === 若需要輸出與原音相同取樣率，則再升頻 ===
    if sr != target_sr:
        enhanced = librosa.resample(enhanced, orig_sr=target_sr, target_sr=orig_sr)
        sr = orig_sr
        print(f"🔁 Resampled enhanced audio back to {sr} Hz")

    # 保證輸入與輸出長度一致（若差 1–2 點可自動裁切）
    if len(enhanced) != len(noisy):
        min_len = min(len(enhanced), len(noisy))
        enhanced = enhanced[:min_len]
        noisy = noisy[:min_len]

    assert len(noisy) == len(enhanced)

    return noisy, enhanced, sr
