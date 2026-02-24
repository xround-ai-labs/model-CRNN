
"""
python infer_crnn.py \
  --input_dir /home/user/hdd/test_waves/vctk_test \
  --output_dir crnn_keras \
  --sample_rate 25000 \
  --backend keras \
  --weights /home/user/Documents/ai-training/model-CRNN/checkpoints_tf/20260212-112502/best.weights.h5

python infer_crnn.py \
  --input_dir /home/user/Documents/ai-training/gtcrn/test_wavs/remixed_XR_24k/ \
  --output_dir crnn_keras \
  --sample_rate 25000 \
  --backend keras \
  --weights /home/user/Documents/ai-training/model-CRNN/checkpoints_tf/20260212-112502/best.weights.h5
  
python infer_crnn.py \
  --input_dir /home/user/Documents/ai-training/gtcrn/test_wavs/remixed_XR_24k/ \
  --output_dir crnn_tflite \
  --sample_rate 16000 \
  --backend tflite \
  --tflite /home/user/Documents/ai-training/model-CRNN/checkpoints_tf/crnn_keras.tflite

sox crnn_keras/remixed_XR_snr_0_24k_1_crnn.wav -n stat -freq

"""

import os
os.environ["TF_LITE_DISABLE_XNNPACK"] = "1"

import argparse
import numpy as np
import soundfile as sf
import librosa
import tensorflow as tf

from model_tf.crn import MiniCRN_Causal128

# ============================================================
# STFT / Chunk parameters (MUST match training exactly)
# ============================================================
N_FFT = 100
HOP = 25
WIN = 100

CHUNK_FRAMES = 200
SEG_SAMPLES = (CHUNK_FRAMES - 1) * HOP + N_FFT  # 5075 samples

_INV_WIN_FN = tf.signal.inverse_stft_window_fn(
    frame_step=HOP,
    forward_window_fn=tf.signal.hann_window
)

# ============================================================
# Audio I/O
# ============================================================
def load_wav_resample(path, target_sr):
    wav, sr = sf.read(path, always_2d=False)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=-1)
    wav = wav.astype(np.float32)
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    return wav

def save_wav(path, wav, sr):
    wav = np.clip(wav, -1.0, 1.0)
    sf.write(path, wav, sr)

# ============================================================
# STFT helpers (waveform <-> mag/phase)
# ============================================================
def wav_to_mag_phase(wav_1d):
    """
    wav_1d: (T,)
    returns:
        mag_4d : (1, 51, 200, 1)
        phase  : (1, 200, 51)
    """
    wav = tf.convert_to_tensor(wav_1d[None, :], tf.float32)

    stft = tf.signal.stft(
        wav,
        frame_length=WIN,
        frame_step=HOP,
        fft_length=N_FFT,
        window_fn=tf.signal.hann_window,
        pad_end=False,
    )  # (1, T, F)

    mag = tf.abs(stft)
    phase = tf.math.angle(stft)

    mag = tf.transpose(mag, [0, 2, 1])[..., None]  # (1, F, T, 1)
    return mag, phase

def mag_phase_to_wav(est_mag_4d, phase):
    """
    est_mag_4d: (1, 51, 200, 1)
    phase:      (1, 200, 51)
    """
    mag = tf.squeeze(est_mag_4d, axis=-1)          # (1, 51, 200)
    mag = tf.transpose(mag, [0, 2, 1])             # (1, 200, 51)

    real = mag * tf.cos(phase)
    imag = mag * tf.sin(phase)
    stft = tf.complex(real, imag)

    wav = tf.signal.inverse_stft(
        stft,
        frame_length=WIN,
        frame_step=HOP,
        fft_length=N_FFT,
        window_fn=_INV_WIN_FN
    )

    return wav.numpy().squeeze()

# ============================================================
# Backend: Keras
# ============================================================
class KerasBackend:
    def __init__(self, weights_path, sample_rate):
        self.model = MiniCRN_Causal128()

        # build model
        dummy = tf.zeros([1, N_FFT // 2 + 1, CHUNK_FRAMES, 1], tf.float32)
        self.model(dummy, training=False)

        self.model.load_weights(weights_path)

    def infer_mag(self, mag_4d):
        # IMPORTANT: training=False, no LSTM states
        return self.model(mag_4d, training=False)

# ============================================================
# Backend: TFLite
# ============================================================
class TFLiteBackend:
    def __init__(self, tflite_path):
        self.interpreter = tf.lite.Interpreter(model_path=tflite_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_shape = tuple(self.input_details[0]["shape"])

    def infer_mag(self, mag_4d):
        x = mag_4d.numpy().astype(np.float32)

        if x.shape != self.input_shape:
            raise ValueError(f"TFLite expects {self.input_shape}, got {x.shape}")

        self.interpreter.set_tensor(self.input_details[0]["index"], x)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_details[0]["index"])
        return tf.convert_to_tensor(out)

# ============================================================
# Core CRN inference pipeline (chunk-based, stateless)
# ============================================================
def run_crn_pipeline(backend, wav_1d):
    """
    EXACTLY matches training assumptions:
    - chunk-based
    - STFT per chunk
    - NO LSTM state carried across chunks
    """
    wav_1d = wav_1d.astype(np.float32)
    orig_len = len(wav_1d)

    hop_samples = SEG_SAMPLES // 2  # 50% overlap
    out = np.zeros(len(wav_1d) + SEG_SAMPLES, dtype=np.float32)
    weight = np.zeros_like(out)

    window = np.hanning(SEG_SAMPLES).astype(np.float32)

    for start in range(0, len(wav_1d), hop_samples):
        end = start + SEG_SAMPLES
        seg = wav_1d[start:end]

        if len(seg) < SEG_SAMPLES:
            seg = np.pad(seg, (0, SEG_SAMPLES - len(seg)))

        mag_4d, phase = wav_to_mag_phase(seg)
        est_mag_4d = backend.infer_mag(mag_4d)
        seg_out = mag_phase_to_wav(est_mag_4d, phase)

        seg_out = seg_out[:SEG_SAMPLES] * window

        out[start:start+SEG_SAMPLES] += seg_out
        weight[start:start+SEG_SAMPLES] += window

    weight[weight == 0] = 1.0
    out /= weight

    return out[:orig_len]

# ============================================================
# Main
# ============================================================
def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    if args.backend == "keras":
        backend = KerasBackend(args.weights, args.sample_rate)
    else:
        backend = TFLiteBackend(args.tflite)

    wav_files = sorted(
        f for f in os.listdir(args.input_dir)
        if f.lower().endswith(".wav")
    )

    for fname in wav_files:
        in_path = os.path.join(args.input_dir, fname)
        print(f"Processing: {fname}")

        wav = load_wav_resample(in_path, args.sample_rate)
        out_wav = run_crn_pipeline(backend, wav)

        out_name = os.path.splitext(fname)[0] + "_crnn.wav"
        out_path = os.path.join(args.output_dir, out_name)
        save_wav(out_path, out_wav, args.sample_rate)

    print("Inference finished.")

# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sample_rate", type=int, default=16000)

    parser.add_argument("--backend", choices=["keras", "tflite"], required=True)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--tflite", default=None)

    args = parser.parse_args()

    if args.backend == "keras" and not args.weights:
        raise ValueError("backend=keras requires --weights")
    if args.backend == "tflite" and not args.tflite:
        raise ValueError("backend=tflite requires --tflite")

    main(args)
