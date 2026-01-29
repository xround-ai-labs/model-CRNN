#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert MiniCRN_Causal128 best.weights.h5 -> TFLite

Usage:
  python export_best_weights_to_tflite.py

Notes:
- This script reconstructs the model, loads Keras weights (.weights.h5),
  and exports a TFLite model.
- If your model contains TensorList/LSTM ops, TFLite may require SELECT_TF_OPS
  (Flex delegate). This script enables it by default.
"""

import os
os.environ["TF_USE_CUDNN"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf

# ---------- Paths ----------
WEIGHTS_PATH = "./checkpoints_tf/20260128-181411/best.weights.h5"
OUT_TFLITE   = "./checkpoints_tf/crnn_keras.tflite"

# ---------- Model / input shape (must match training) ----------
# From your training script:
SAMPLE_RATE = 16000
N_FFT = 100
CHUNK_FRAMES = 200

# TF STFT magnitude used in training produces F = N_FFT//2 + 1
F_BINS = N_FFT // 2 + 1

# ---------- Import your model definition ----------
# This must be resolvable in PYTHONPATH (same project root as training).
from model_tf import MiniCRN_Causal128


def build_model():
    model = MiniCRN_Causal128(n_fft=N_FFT)
    # Build by running a dummy forward pass (same as training)
    dummy = tf.zeros([1, F_BINS, CHUNK_FRAMES, 1], dtype=tf.float32)
    _ = model(dummy, None, training=False)  # model returns (pred, states)
    return model


def main():
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"WEIGHTS_PATH not found: {WEIGHTS_PATH}")

    # 1) Build + load weights
    model = build_model()
    model.load_weights(WEIGHTS_PATH)
    print(f"[OK] Loaded weights: {WEIGHTS_PATH}")

    # 2) Create a serving function for TFLite conversion
    # We export a function that takes only the magnitude input and returns pred
    @tf.function(
        input_signature=[
            tf.TensorSpec([1, F_BINS, CHUNK_FRAMES, 1], tf.float32, name="noisy_mag")
        ]
    )
    def serve(noisy_mag):
        pred, _ = model(noisy_mag, None, training=False)
        return {"pred": pred}

    concrete_fn = serve.get_concrete_function()

    # 3) Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn])

    # Recommended for many CRNN/LSTM-style graphs:
    # - Enable SELECT_TF_OPS if TensorList ops appear (Flex delegate requirement)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]

    # If you previously dealt with TensorList lowering issues, you can toggle this:
    # - True: try to lower TensorList ops (sometimes helps, sometimes breaks)
    # - False: keep TensorList (often forces SELECT_TF_OPS)
    converter._experimental_lower_tensor_list_ops = False

    # Optional: basic optimization (keeps float32 unless you add quantization)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(OUT_TFLITE), exist_ok=True)
    with open(OUT_TFLITE, "wb") as f:
        f.write(tflite_model)

    print(f"[OK] Wrote TFLite: {OUT_TFLITE}")
    print("[Info] If runtime warns about Flex delegate, that's expected when using SELECT_TF_OPS.")


if __name__ == "__main__":
    main()
