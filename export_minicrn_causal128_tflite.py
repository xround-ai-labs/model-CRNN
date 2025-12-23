"""
conda create -n tf2tflite python=3.10 -y
conda activate tf2tflite

pip install \
  tensorflow==2.15.0 \
  numpy==1.26.4

python export_minicrn_causal128_tflite.py
"""


import tensorflow as tf

def main():
    converter = tf.lite.TFLiteConverter.from_saved_model(
        "MiniCRN_Causal128_tf"
    )

    # ★ LSTM 必須允許 SELECT_TF_OPS
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]

    # ★ LSTM 使用 resource variables
    converter.experimental_enable_resource_variables = True

    # （先不要量化，確保能轉）
    tflite_model = converter.convert()

    out_path = "MiniCRN_Causal128.tflite"
    with open(out_path, "wb") as f:
        f.write(tflite_model)

    print("✅ TFLite exported: MiniCRN_Causal128.tflite")

if __name__ == "__main__":
    main()
