"""
conda create -n onnx python=3.10 -y
conda activate onnx

pip install torch==2.1.2
pip install onnx==1.14.1
pip install onnx-tf==1.10.0

pip install torch==2.3.0
pip install onnx==1.16.0
pip install onnx-tf==1.10.0
pip install tensorflow==2.15.0
pip install numpy
"""

"""
TORCH_ONNX_USE_EXPERIMENTAL_EXPORTER=0 python export_minicrn_causal128_onnx.py

onnx-tf convert \
  -i MiniCRN_Causal128.onnx \
  -o MiniCRN_Causal128_tf
"""

import torch
from model.crn import MiniCRN_Causal128

def main():
    model = MiniCRN_Causal128(n_fft=100)
    model.eval()

    # 固定輸入 shape（TFLite 不支援真正 dynamic）
    dummy = torch.randn(1, 1, 51, 200)

    torch.onnx.export(
        model,
        dummy,
        "MiniCRN_Causal128.onnx",
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
        do_constant_folding=True,
        dynamic_axes=None,
    )

    print("✅ ONNX exported: MiniCRN_Causal128.onnx")

if __name__ == "__main__":
    main()
