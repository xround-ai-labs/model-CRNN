import os
import time
from pathlib import Path

import json5
import torch
from torch.utils.data import DataLoader

from util.utils import initialize_config, prepare_device, prepare_empty_dir


class BaseInferencer:
    def __init__(self, config, checkpoint_path, output_dir):
        checkpoint_path = Path(checkpoint_path).expanduser().absolute()
        output_root_dir = Path(output_dir).expanduser().absolute()
        self.device = prepare_device(torch.cuda.device_count())

        self.enhanced_dir = output_root_dir
        prepare_empty_dir([self.enhanced_dir])

        self.dataloader = self._load_dataloader(config["dataset"])
        self.model = self._load_model(config["model"], checkpoint_path, self.device)
        self.inference_config = config["inference"]

        print("Configurations are as follows: ")
        print(json5.dumps(config, indent=2, sort_keys=False))

        with open((output_root_dir / f"{time.strftime('%Y-%m-%d-%H-%M-%S')}.json").as_posix(), "w") as handle:
            json5.dump(config, handle, indent=2, sort_keys=False)

    @staticmethod
    def _load_dataloader(dataset_config):
        dataset = initialize_config(dataset_config)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=1,
            num_workers=0,
        )
        return dataloader

    @staticmethod
    def _load_model(model_config, checkpoint_path, device):
        model = initialize_config(model_config).to(device)

        # === 先用 dummy forward 初始化 LSTM 結構 ===
        try:
            with torch.no_grad():
                dummy = torch.zeros(1, 1, 51, 10, device=device)
                _ = model(dummy)
            print("✅ 模型結構已初始化（LSTM 已建立）")
        except Exception as e:
            print(f"⚠️ Dummy forward 初始化失敗（可忽略）: {e}")

        # === 載入 checkpoint ===
        if os.path.splitext(os.path.basename(checkpoint_path))[-1] == ".tar":
            model_checkpoint = torch.load(checkpoint_path, map_location=device)
            model_static_dict = model_checkpoint["model"]
            print(f"Loading model checkpoint with *.tar format, the epoch is: {model_checkpoint['epoch']}.")
        else:
            model_static_dict = torch.load(checkpoint_path, map_location=device)

        # === 正式載入權重 ===
        model.load_state_dict(model_static_dict)
        model.eval()
        print(f"✅ 模型權重載入完成：{checkpoint_path}")
        return model


    def inference(self):
        raise NotImplementedError
