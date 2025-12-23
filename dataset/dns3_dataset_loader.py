import os
import csv
from dataset.dataset_variable_length_input import Dataset
from torch.utils.data import DataLoader

# ========================================
# CSV paths (DNS3)
# ========================================
train_pairs_data_csv = '/home/user/hdd/dataset_download_scripts/datasets_synthesized/0307/pairs_csv/dns3_1819hrs_train_pairs_add_quality_check.csv'
val_pairs_data_csv   = '/home/user/hdd/dataset_download_scripts/datasets_synthesized/0307/pairs_csv/dns3_1819hrs_val_pairs_add_quality_check.csv'

# Set the number of training data to use, if 0 use all
train_num = 0

# ========================================
# Utility: read CSV and filter pairs
# ========================================
def make_pair_list_from_csv(csv_path, output_txt):
    """
    CSV format:
        col 0: clean wav path
        col 1: noisy wav path
        col 2: is_reverb (TRUE/FALSE)
        col 6: clean_voice_residual_snr
    Conditions:
        - is_reverb == FALSE
        - clean_voice_residual_snr > 30
    """
    valid_pairs = 0

    with open(csv_path, 'r', newline='', encoding='utf-8') as f, \
         open(output_txt, 'w') as out_f:

        reader = csv.reader(f)
        header = next(reader)  # skip header

        for row in reader:
            try:
                clean_wav = row[0]
                noisy_wav = row[1]
                is_reverb = row[2].strip().upper()
                clean_snr = float(row[6])

                if is_reverb != 'FALSE':
                    continue
                if clean_snr <= 30:
                    continue
                if not (os.path.exists(clean_wav) and os.path.exists(noisy_wav)):
                    continue

                out_f.write(f"{noisy_wav} {clean_wav}\n")
                valid_pairs += 1

            except Exception as e:
                print(f"⚠️ Skip row due to error: {e}")
                continue

    print(f"✅ Pair list saved to: {output_txt} ({valid_pairs} pairs)")

# ========================================
# Generate training / validation pair lists
# ========================================
os.makedirs("./dataset_lists", exist_ok=True)

train_list_txt = "./dataset_lists/train_dns3_pairs.txt"
val_list_txt   = "./dataset_lists/val_dns3_pairs.txt"

make_pair_list_from_csv(train_pairs_data_csv, train_list_txt)
make_pair_list_from_csv(val_pairs_data_csv, val_list_txt)

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

train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# ========================================
# Quick check
# ========================================
if __name__ == "__main__":
    for noisy_mag, clean_mag, length, name in train_loader:
        print(
            f"Batch file: {name[0]}, "
            f"noisy shape: {noisy_mag.shape}, "
            f"clean shape: {clean_mag.shape}"
        )
        break
