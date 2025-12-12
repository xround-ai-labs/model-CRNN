import argparse
import json5
import torch
from util.utils import initialize_config

def main(config, checkpoint_path, output_dir):
    inferencer_class = initialize_config(config["inference"], pass_args=False)

    # === 嘗試預先建立模型以避免 LSTM 未初始化 ===
    # 若 config["model"] 存在，可以從這裡初始化模型
    if "model" in config:
        model_class = initialize_config(config["model"], pass_args=False)
        model = model_class()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Dummy forward 初始化 LSTM 結構
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 51, 10, device=device)
            _ = model(dummy)
        print("✅ 模型結構已初始化，LSTM 已建立。")

    # === 建立推論器 ===
    inferencer = inferencer_class(
        config,
        checkpoint_path,
        output_dir
    )

    inferencer.inference()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Inference / Enhancement / Test")
    parser.add_argument("-C", "--config", type=str, required=True, help="Inference config file.")
    parser.add_argument("-cp", "--checkpoint_path", type=str, required=True, help="Model checkpoint")
    parser.add_argument("-dist", "--output_dir", type=str, required=True, help="The dir of saving enhanced wav files.")
    args = parser.parse_args()

    config = json5.load(open(args.config))
    main(config, args.checkpoint_path, args.output_dir)
