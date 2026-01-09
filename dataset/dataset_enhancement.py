import os

import librosa
from torch.utils import data
import soundfile as sf


class Dataset(data.Dataset):
    def __init__(self,
                 noisy_dataset,
                 limit,
                 offset,
                 sr,
                 ):
        """
        Args:
            noisy_dataset (str): noisy dir (wav format files) or noisy filenames list
        """
        noisy_dataset = os.path.abspath(os.path.expanduser(noisy_dataset))

        if os.path.isfile(noisy_dataset):
            noisy_wav_files = [line.rstrip('\n') for line in open(os.path.abspath(os.path.expanduser(noisy_dataset)), "r")]
            if offset:
                noisy_wav_files = noisy_wav_files[offset:]
            if limit:
                noisy_wav_files = noisy_wav_files[:limit]
        elif os.path.isdir(noisy_dataset):
            noisy_wav_files = librosa.util.find_files(noisy_dataset, ext="wav", limit=limit, offset=offset)
        else:
            raise FileNotFoundError(f"Please Check {noisy_dataset}")

        print(f"Number of noisy files in the dir {noisy_dataset}: {len(noisy_wav_files)}")

        self.length = len(noisy_wav_files)
        self.noisy_wav_files = noisy_wav_files
        self.sr = sr

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        noisy_path = self.noisy_wav_files[item]

        # 只讀 header，拿到真實 sr
        info = sf.info(noisy_path)
        sr = info.samplerate

        # 讀 waveform（不重採樣）
        noisy, _ = librosa.load(noisy_path, sr=None)

        name = os.path.splitext(os.path.basename(noisy_path))[0]

        # 回傳「路徑」或「sr」，Inferencer 才能判斷
        return noisy, noisy_path, sr
