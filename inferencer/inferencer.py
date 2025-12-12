import os
import soundfile as sf
import torch
from tqdm import tqdm
import librosa

from inferencer.base_inferencer import BaseInferencer
from inferencer.inferencer_full_band import full_band_no_truncation


@torch.no_grad()
def inference_wrapper(
        dataloader,
        model,
        device,
        inference_args,
        enhanced_dir
):
    for noisy, name in tqdm(dataloader, desc="Inference"):
        assert len(name) == 1, "The batch size of inference stage must be 1."
        name = name[0]

        # === noisy 是 Tensor，轉為 numpy ===
        noisy = noisy.squeeze().cpu().numpy()

        # 嘗試自動偵測取樣率（若 dataloader 無法提供，就假設 16k）
        try:
            sr = librosa.get_samplerate(name)
            print(f"🎧 Detected samplerate from file name: {sr}")
        except Exception:
            sr = 16000
            print(f"⚠️  Cannot detect samplerate from dataloader; using default sr={sr}")

        # === 模型推論 ===
        if inference_args["inference_type"] == "full_band_no_truncation":
            noisy, enhanced, sr = full_band_no_truncation(model, device, inference_args, noisy, sr)
        else:
            raise NotImplementedError(f"Not implemented Inferencer type: {inference_args['inference_type']}")

        # === 建立輸出檔名 ===
        base_name = os.path.splitext(name)[0]
        enhanced_filename = f"{base_name}_crnn.wav"
        output_path = enhanced_dir / enhanced_filename

        # === 寫出檔案 ===
        sf.write(output_path, enhanced, sr)
        print(f"✅ Saved: {output_path} ({sr} Hz)")


class Inferencer(BaseInferencer):
    def __init__(self, config, checkpoint_path, output_dir):
        super(Inferencer, self).__init__(config, checkpoint_path, output_dir)

    @torch.no_grad()
    def inference(self):
        inference_wrapper(
            dataloader=self.dataloader,
            model=self.model,
            device=self.device,
            inference_args=self.inference_config,
            enhanced_dir=self.enhanced_dir
        )
