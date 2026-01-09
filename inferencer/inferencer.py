import os
import soundfile as sf
import torch
from tqdm import tqdm
import librosa

from inferencer.base_inferencer import BaseInferencer
from inferencer.inferencer_full_band import full_band_no_truncation
from inferencer.inferencer_streaming_full_band import streaming_full_band
from inferencer.tflite_full_band_no_truncation import tflite_full_band_no_truncation

@torch.no_grad()
def inference_wrapper(
        dataloader,
        model,
        device,
        inference_args,
        enhanced_dir,
        checkpoint_path
):
    for noisy, path, sr in tqdm(dataloader, desc="Inference"):

        path = path[0]
        sr = int(sr[0])

        noisy = noisy.squeeze().cpu().numpy()

        print(f"🎧 Input samplerate: {sr} Hz ({path})")

        # === 模型推論 ===
        if inference_args["inference_type"] == "full_band_no_truncation":
            # PyTorch full-band
            noisy, enhanced, sr = full_band_no_truncation(
                model, device, inference_args, noisy, sr
            )

        elif inference_args["inference_type"] == "streaming_full_band":
            # PyTorch streaming
            enhanced, sr = streaming_full_band(
                model, device, inference_args, noisy, sr
            )

        elif inference_args["inference_type"] == "tflite_full_band_no_truncation":
            # ✅ TFLite full-band（不使用 PyTorch model / device）
            tflite_model_path = checkpoint_path
            noisy, enhanced, sr = tflite_full_band_no_truncation(
                tflite_model_path=tflite_model_path,
                inference_args=inference_args,
                noisy=noisy,
                sr=sr,
            )

        else:
            raise NotImplementedError(
                f"Not implemented inference_type: {inference_args['inference_type']}"
            )

        # === 建立輸出檔名 ===
        base_name = os.path.splitext(os.path.basename(path))[0]
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
            enhanced_dir=self.enhanced_dir,
            checkpoint_path=self.checkpoint_path
        )
