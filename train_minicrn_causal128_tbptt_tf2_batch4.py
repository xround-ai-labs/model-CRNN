
# Optimized MiniCRN_Causal128 training script (batched chunks)
# Focus: fast convergence + GPU utilization with chunk-level batching and batched STFT
#
# Run:
#   python train_minicrn_causal128_tbptt_tf2_batch4.py
#
# Notes:
# - This uses CHUNK-level batching (independent chunks). LSTM states are NOT carried across chunks in a batch.
#   This is the standard trade-off to unlock GPU throughput (batching) while keeping streaming-sized chunks.

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import time
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import soundfile as sf
import resampy

from model_tf import MiniCRN_Causal128

# ===================== Streaming / TBPTT (chunk size in STFT frames) =====================
CHUNK_FRAMES = 200
CHUNK_HOP_FRAMES = 50
LOSS_FRAMES = CHUNK_HOP_FRAMES

# In chunk-batched training, per-chunk state warm-up is not meaningful (state is reset per chunk).
WARMUP_CHUNKS = 0

# Per-utterance: how many chunks we sample (speed/coverage trade-off)
TRAIN_CHUNKS_PER_UTT = 16

# ===================== Chunk batching =====================
BATCH_SIZE = 4  # requested

# ===================== Dataset ===============================
TRAIN_LIST_TXT = "./dataset_lists/train_vctk_pairs.txt"
VAL_LIST_TXT   = "./dataset_lists/val_vctk_pairs.txt"

SAMPLE_RATE = 16000
N_FFT = 100
HOP_LENGTH = 25

# waveform length needed to produce exactly CHUNK_FRAMES STFT frames
SEG_SAMPLES = (CHUNK_FRAMES - 1) * HOP_LENGTH + N_FFT
# waveform stride between adjacent candidate chunks
HOP_SAMPLES = CHUNK_HOP_FRAMES * HOP_LENGTH

# ===================== Training ==============================
EPOCHS = 50
LEARNING_RATE = 1e-3
GRAD_CLIP_NORM = 5.0

# === checkpoint root ===
CKPT_ROOT = "./checkpoints_tf"
os.makedirs(CKPT_ROOT, exist_ok=True)

RUN_ID = time.strftime("%Y%m%d-%H%M%S")
CKPT_DIR = os.path.join(CKPT_ROOT, RUN_ID)
os.makedirs(CKPT_DIR, exist_ok=True)
print(f"[Checkpoint] saving to: {CKPT_DIR}")

# ===================== TensorBoard ==========================
TB_RUN_DIR = os.path.join(CKPT_DIR, "log")
os.makedirs(TB_RUN_DIR, exist_ok=True)

tb_train_writer = tf.summary.create_file_writer(os.path.join(TB_RUN_DIR, "train"))
tb_val_writer   = tf.summary.create_file_writer(os.path.join(TB_RUN_DIR, "val"))

print(f"[TensorBoard] logdir: {TB_RUN_DIR}")


# ===================== Utilities =============================
def read_pair_list(txt_path):
    pairs = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                noisy, clean = line.strip().split()
                pairs.append((noisy, clean))
    return pairs

def load_wav_resample(path, target_sr):
    wav, sr = sf.read(path, always_2d=False)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=-1)
    wav = wav.astype(np.float32)
    if sr != target_sr:
        wav = resampy.resample(wav, sr, target_sr).astype(np.float32)
    return wav

def random_chunk_starts(num_samples, seg_samples, hop_samples, n_chunks, rng):
    """Return a list of waveform start indices aligned to hop_samples."""
    if num_samples < seg_samples:
        return []
    max_start = num_samples - seg_samples
    max_k = max_start // hop_samples
    if max_k < 0:
        return []
    population = max_k + 1
    k = min(n_chunks, population)
    ks = rng.choice(population, size=k, replace=False)
    return [int(x) * hop_samples for x in ks]

def stft_mag_tf(wav_batch):
    """
    wav_batch: [B, S] float32
    returns: [B, F, T, 1] where T=CHUNK_FRAMES
    """
    stft = tf.signal.stft(
        wav_batch,
        frame_length=N_FFT,
        frame_step=HOP_LENGTH,
        fft_length=N_FFT,
        window_fn=tf.signal.hann_window,
        pad_end=False,
    )  # [B, T, F]
    mag = tf.abs(stft)                  # [B, T, F]
    mag = tf.transpose(mag, [0, 2, 1])  # [B, F, T]
    return mag[..., tf.newaxis]         # [B, F, T, 1]

def log1p_mse(pred, target):
    pred = tf.maximum(pred, 0.0)
    target = tf.maximum(target, 0.0)
    return tf.reduce_mean(tf.square(tf.math.log1p(pred) - tf.math.log1p(target)))

def loss_on_latest_frames(pred, target):
    return log1p_mse(pred[:, :, -LOSS_FRAMES:, :], target[:, :, -LOSS_FRAMES:, :])

# ===================== Model ================================
model = MiniCRN_Causal128(n_fft=N_FFT)
dummy = tf.zeros([1, N_FFT // 2 + 1, CHUNK_FRAMES, 1], tf.float32)
model(dummy, training=False)
del dummy

optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

@tf.function
def train_step_batched(noisy_mag, clean_mag):
    """
    Batched chunk training step.
    noisy_mag/clean_mag: [B, F, T, 1]
    LSTM states are reset per chunk (states=None) to enable batching.
    """
    with tf.GradientTape() as tape:
        pred, _ = model(noisy_mag, None, training=False)
        loss = loss_on_latest_frames(pred, clean_mag)
    grads = tape.gradient(loss, model.trainable_variables)
    grad_norm = tf.linalg.global_norm([g for g in grads if g is not None])
    grads = [tf.clip_by_norm(g, GRAD_CLIP_NORM) if g is not None else None for g in grads]
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, grad_norm

def eval_utterance_loss(noisy, clean, n_chunks=8, seed=0):
    """
    Lightweight validation: sample a few chunks; evaluate in batches for speed.
    """
    L = min(len(noisy), len(clean))
    if L < SEG_SAMPLES:
        return None
    noisy = noisy[:L]
    clean = clean[:L]

    rngv = np.random.default_rng(seed)
    starts = random_chunk_starts(L, SEG_SAMPLES, HOP_SAMPLES, n_chunks, rngv)
    if not starts:
        return None

    # build batches
    vlosses = []
    bn, bc = [], []
    for s in starts:
        bn.append(noisy[s:s+SEG_SAMPLES])
        bc.append(clean[s:s+SEG_SAMPLES])

        if len(bn) == BATCH_SIZE:
            n_tf = tf.convert_to_tensor(np.stack(bn, axis=0), tf.float32)
            c_tf = tf.convert_to_tensor(np.stack(bc, axis=0), tf.float32)
            n_mag = stft_mag_tf(n_tf)
            c_mag = stft_mag_tf(c_tf)
            pred, _ = model(n_mag, None, training=False)
            v = float(loss_on_latest_frames(pred, c_mag).numpy())
            vlosses.append(v)
            bn, bc = [], []

    # tail
    if len(bn) > 0:
        n_tf = tf.convert_to_tensor(np.stack(bn, axis=0), tf.float32)
        c_tf = tf.convert_to_tensor(np.stack(bc, axis=0), tf.float32)
        n_mag = stft_mag_tf(n_tf)
        c_mag = stft_mag_tf(c_tf)
        pred, _ = model(n_mag, None, training=False)
        v = float(loss_on_latest_frames(pred, c_mag).numpy())
        vlosses.append(v)

    return float(np.mean(vlosses)) if vlosses else None

# ===================== Training loop =========================
train_pairs = read_pair_list(TRAIN_LIST_TXT)
val_pairs = read_pair_list(VAL_LIST_TXT)

rng = np.random.default_rng(0)
best_val = float("inf")

print(f"Train pairs: {len(train_pairs)} | Val pairs: {len(val_pairs)}")
print(f"SEG_SAMPLES={SEG_SAMPLES} | HOP_SAMPLES={HOP_SAMPLES} | BATCH_SIZE={BATCH_SIZE}")

for epoch in range(1, EPOCHS + 1):
    rng.shuffle(train_pairs)
    losses = []

    # buffers for batched chunks
    batch_noisy = []
    batch_clean = []

    for noisy_p, clean_p in tqdm(train_pairs, desc=f"Epoch {epoch}"):
        noisy = load_wav_resample(noisy_p, SAMPLE_RATE)
        clean = load_wav_resample(clean_p, SAMPLE_RATE)
        L = min(len(noisy), len(clean))
        if L < SEG_SAMPLES:
            continue

        starts = random_chunk_starts(
            L, SEG_SAMPLES, HOP_SAMPLES,
            TRAIN_CHUNKS_PER_UTT + WARMUP_CHUNKS, rng
        )
        if not starts:
            continue

        for s in starts:
            n_seg = noisy[s:s+SEG_SAMPLES]
            c_seg = clean[s:s+SEG_SAMPLES]
            batch_noisy.append(n_seg)
            batch_clean.append(c_seg)

            if len(batch_noisy) == BATCH_SIZE:
                n_tf = tf.convert_to_tensor(np.stack(batch_noisy, axis=0), tf.float32)  # [B, S]
                c_tf = tf.convert_to_tensor(np.stack(batch_clean, axis=0), tf.float32)
                n_mag = stft_mag_tf(n_tf)  # [B, F, T, 1]
                c_mag = stft_mag_tf(c_tf)

                loss, grad_norm = train_step_batched(n_mag, c_mag)
                losses.append(float(loss.numpy()))

                batch_noisy.clear()
                batch_clean.clear()

    # flush remaining partial batch
    if len(batch_noisy) > 0:
        n_tf = tf.convert_to_tensor(np.stack(batch_noisy, axis=0), tf.float32)
        c_tf = tf.convert_to_tensor(np.stack(batch_clean, axis=0), tf.float32)
        n_mag = stft_mag_tf(n_tf)
        c_mag = stft_mag_tf(c_tf)
        loss, grad_norm = train_step_batched(n_mag, c_mag)
        losses.append(float(loss.numpy()))
        batch_noisy.clear()
        batch_clean.clear()

    if len(losses) == 0:
        print(f"[Warning] Epoch {epoch}: no valid training steps; skipping checkpoints.")
        continue

    train_loss = float(np.mean(losses))

    # ===== lightweight validation (subset) =====
    VAL_MAX_UTTS = min(50, len(val_pairs))
    VAL_CHUNKS_PER_UTT = 8
    vls = []
    for noisy_p, clean_p in val_pairs[:VAL_MAX_UTTS]:
        noisy_v = load_wav_resample(noisy_p, SAMPLE_RATE)
        clean_v = load_wav_resample(clean_p, SAMPLE_RATE)
        v = eval_utterance_loss(noisy_v, clean_v, n_chunks=VAL_CHUNKS_PER_UTT, seed=epoch)
        if v is not None and np.isfinite(v):
            vls.append(v)
    val_loss = float(np.mean(vls)) if len(vls) > 0 else float("nan")

    print(f"Epoch {epoch} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

    # ===== TensorBoard logging =====
    with tb_train_writer.as_default():
        tf.summary.scalar("loss", train_loss, step=epoch)
        tf.summary.scalar("lr", tf.convert_to_tensor(optimizer.learning_rate), step=epoch)
        tf.summary.scalar("grad/global_norm", grad_norm, step=epoch)

    if np.isfinite(val_loss):
        with tb_val_writer.as_default():
            tf.summary.scalar("loss", val_loss, step=epoch)

    # ===== epoch checkpoint =====
    epoch_ckpt_path = os.path.join(CKPT_DIR, f"epoch{epoch:03d}.weights.h5")
    model.save_weights(epoch_ckpt_path)
    print(f"[Checkpoint] saved epoch {epoch}: {epoch_ckpt_path}")

    # ===== best checkpoint (by val) =====
    if np.isfinite(val_loss) and val_loss < best_val:
        best_val = val_loss
        best_ckpt_path = os.path.join(CKPT_DIR, "best.weights.h5")
        model.save_weights(best_ckpt_path)
        print(f"[Checkpoint] new BEST at epoch {epoch} (val_loss={best_val:.6f})")


print("Training finished")
