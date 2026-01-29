import numpy as np
import librosa
import tensorflow as tf


def tflite_full_band_no_truncation(
    tflite_model_path,
    inference_args,
    noisy,
    sr=24000,
):
    n_fft = inference_args["n_fft"]
    hop_length = inference_args["hop_length"]
    win_length = inference_args["win_length"]
    target_sr = 24000

    BLOCK_T = 200
    HOP_T = 100

    # ===== resample =====
    if sr != target_sr:
        noisy = librosa.resample(noisy, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # ===== STFT =====
    noisy_stft = librosa.stft(
        noisy,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
    )
    noisy_mag, noisy_phase = librosa.magphase(noisy_stft)

    F, T = noisy_mag.shape

    # ===== pad to fit blocks =====
    pad_T = (BLOCK_T - (T - BLOCK_T) % HOP_T) % HOP_T
    noisy_mag = np.pad(noisy_mag, ((0, 0), (0, pad_T)), mode="constant")
    noisy_phase = np.pad(noisy_phase, ((0, 0), (0, pad_T)), mode="constant")

    total_T = noisy_mag.shape[1]

    # ===== TFLite init =====
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    in_idx = interpreter.get_input_details()[0]["index"]
    interpreter.resize_tensor_input(in_idx, [1, 1, F, BLOCK_T], strict=True)
    interpreter.allocate_tensors()

    out_idx = interpreter.get_output_details()[0]["index"]

    # ===== overlap-add buffers =====
    enhanced_mag = np.zeros((F, total_T), dtype=np.float32)
    weight = np.zeros((total_T,), dtype=np.float32)

    window = np.hanning(BLOCK_T).astype(np.float32)

    # ===== block inference =====
    for t0 in range(0, total_T - BLOCK_T + 1, HOP_T):
        block = noisy_mag[:, t0:t0 + BLOCK_T]
        block = block[None, None, :, :].astype(np.float32)

        interpreter.set_tensor(in_idx, block)
        interpreter.invoke()
        out = interpreter.get_tensor(out_idx)[0, 0]  # [F, BLOCK_T]

        enhanced_mag[:, t0:t0 + BLOCK_T] += out * window
        weight[t0:t0 + BLOCK_T] += window

    enhanced_mag /= np.maximum(weight, 1e-8)

    # ===== remove padding =====
    enhanced_mag = enhanced_mag[:, :T]
    noisy_phase = noisy_phase[:, :T]

    # ===== ISTFT =====
    enhanced = librosa.istft(
        enhanced_mag * noisy_phase,
        hop_length=hop_length,
        win_length=win_length,
        length=len(noisy),
    )

    return noisy, enhanced, sr
