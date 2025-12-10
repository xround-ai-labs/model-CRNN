# vctk_dataset_loader.py
import os
import glob
from dataset.dataset_variable_length_input import Dataset
from torch.utils.data import DataLoader

# ========================================
# Dataset paths
# ========================================
train_src_dir = '/home/user/hdd/datasets_vctk/VTCK_Demand/noisy_trainset_56spk_wav/noisy_trainset_56spk_wav'
train_tgt_dir = '/home/user/hdd/datasets_vctk/VTCK_Demand/clean_trainset_56spk_wav/clean_trainset_56spk_wav'
val_src_dir   = '/home/user/hdd/datasets_vctk/VTCK_Demand/noisy_testset_wav'
val_tgt_dir   = '/home/user/hdd/datasets_vctk/VTCK_Demand/clean_testset_wav'

train_pairs_data_csv = ''
val_pairs_data_csv   = ''

# Set the number of training data to use, if train_num is 0, use all the training data
train_num = 0

# ========================================
# Utility: generate noisy/clean path pairs
# ========================================
def make_pair_list(noisy_dir, clean_dir, output_txt):
    noisy_files = sorted(glob.glob(os.path.join(noisy_dir, "**/*.wav"), recursive=True))
    clean_files = sorted(glob.glob(os.path.join(clean_dir, "**/*.wav"), recursive=True))

    if len(noisy_files) != len(clean_files):
        print(f"⚠️ Warning: {len(noisy_files)} noisy vs {len(clean_files)} clean files")

    with open(output_txt, "w") as f:
        for n_path in noisy_files:
            name = os.path.basename(n_path)
            c_path = os.path.join(clean_dir, name)
            if os.path.exists(c_path):
                f.write(f"{n_path} {c_path}\n")

    print(f"✅ Pair list saved to: {output_txt}  ({len(noisy_files)} pairs)")

# ========================================
# Generate training / validation pair lists
# ========================================
os.makedirs("./dataset_lists", exist_ok=True)
train_list_txt = "./dataset_lists/train_vctk_pairs.txt"
val_list_txt   = "./dataset_lists/val_vctk_pairs.txt"

make_pair_list(train_src_dir, train_tgt_dir, train_list_txt)
make_pair_list(val_src_dir, val_tgt_dir, val_list_txt)

# ========================================
# Create Dataset & DataLoader
# ========================================
sample_rate = 16000
n_fft = 100
hop_length = 25

train_dataset = Dataset(
    dataset_list=train_list_txt,
    limit=train_num,
    offset=0,
    sr=sample_rate,
    n_fft=n_fft,
    hop_length=hop_length,
    train=True
)

val_dataset = Dataset(
    dataset_list=val_list_txt,
    limit=0,
    offset=0,
    sr=sample_rate,
    n_fft=n_fft,
    hop_length=hop_length,
    train=False
)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# ========================================
# Quick check
# ========================================
if __name__ == "__main__":
    for noisy_mag, clean_mag, length, name in train_loader:
        print(f"Batch file: {name[0]}, noisy shape: {noisy_mag.shape}, clean shape: {clean_mag.shape}")
        break
