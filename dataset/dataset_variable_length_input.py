import os

import librosa
import numpy as np
from torch.utils import data


class Dataset(data.Dataset):
    def __init__(
            self,
            dataset_list,
            limit,
            offset,
            sr,
            n_fft,
            hop_length,
            train
    ):
        """
        dataset_list(*.txt):
            <noisy_path> <clean_path>\n
        """
        super(Dataset, self).__init__()
        self.sr = sr
        self.train = train

        dataset_list = [
            line.rstrip('\n')
            for line in open(os.path.abspath(os.path.expanduser(dataset_list)), "r")
        ]
        dataset_list = dataset_list[offset:]
        if limit:
            dataset_list = dataset_list[:limit]

        self.dataset_list = dataset_list
        self.length = len(self.dataset_list)
        self.n_fft = n_fft
        self.hop_length = hop_length

    def __len__(self):
        return self.length

    def _load_and_resample(self, wav_path):
        """
        Load wav and resample to target sample rate (self.sr)
        """
        wav, orig_sr = librosa.load(
            os.path.abspath(os.path.expanduser(wav_path)),
            sr=None,          # ⬅️ 關鍵：先用原始取樣率讀
            mono=True
        )

        # 如果原始取樣率不同，才進行 resample
        if orig_sr != self.sr:
            wav = librosa.resample(
                wav,
                orig_sr=orig_sr,
                target_sr=self.sr,
                res_type="kaiser_best"
            )

        return wav.astype(np.float32)

    def __getitem__(self, item):
        noisy_path, clean_path = self.dataset_list[item].split(" ")
        name = os.path.splitext(os.path.basename(noisy_path))[0]

        # ✅ 統一在這裡做重取樣
        noisy = self._load_and_resample(noisy_path)
        clean = self._load_and_resample(clean_path)

        if self.train:
            noisy_stft = librosa.stft(
                noisy,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft
            )
            clean_stft = librosa.stft(
                clean,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft
            )

            noisy_mag, _ = librosa.magphase(noisy_stft)
            clean_mag, _ = librosa.magphase(clean_stft)

            # noisy_mag / clean_mag shape: [F, T]
            return noisy_mag, clean_mag, noisy_mag.shape[-1], name

        else:
            # validation / inference：回傳 waveform（已 resample）
            return noisy, clean, name
