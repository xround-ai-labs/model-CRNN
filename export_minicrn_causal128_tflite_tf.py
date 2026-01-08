"""
使用 TensorFlow 原生模型導出 TFLite

這個腳本使用 Keras 原生的 MiniCRN_Causal128 模型，
直接導出 TFLite，避免 PyTorch → ONNX → TFLite 的轉換問題。

執行方式:
    conda activate tf2tflite
    python export_minicrn_causal128_tflite_tf.py
"""

import os
import sys
import numpy as np
import tensorflow as tf

# 加入專案路徑
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_tf import MiniCRN_Causal128


def main():
    print("=" * 60)
    print("TensorFlow 原生模型 → TFLite 導出")
    print("=" * 60)
    print(f"TensorFlow 版本: {tf.__version__}")
    
    # 模型參數
    n_fft = 100
    freq_bins = n_fft // 2 + 1  # 51
    time_steps = 200
    
    # 建立模型
    print("\n1. 建立模型...")
    model = MiniCRN_Causal128(n_fft=n_fft)
    
    # Build 模型
    dummy_input = tf.random.normal([1, freq_bins, time_steps, 1])
    model(dummy_input, training=False)
    
    print(f"   模型參數量: {model.count_params():,}")
    
    # 建立用於 TFLite 的 Concrete Function
    print("\n2. 建立 Concrete Function...")
    
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[1, freq_bins, time_steps, 1], dtype=tf.float32)
    ])
    def inference_fn(x):
        """TFLite 推論函數 (不回傳 LSTM state)"""
        out, _ = model(x, training=False)
        return out
    
    concrete_func = inference_fn.get_concrete_function()
    
    # 轉換為 TFLite
    print("\n3. 轉換為 TFLite...")
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    
    # 設定轉換選項 - 只使用內建 ops，不需要 SELECT_TF_OPS
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
    ]
    
    # 可選：量化設定
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # converter.target_spec.supported_types = [tf.float16]  # FP16 量化
    
    tflite_model = converter.convert()
    
    # 儲存 TFLite 模型
    output_path = "MiniCRN_Causal128_tf.tflite"
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"\n✅ TFLite 模型已儲存: {output_path}")
    print(f"   模型大小: {len(tflite_model) / 1024:.2f} KB")
    
    # 驗證 TFLite 模型
    print("\n4. 驗證 TFLite 模型...")
    interpreter = tf.lite.Interpreter(model_path=output_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"   Input: {input_details[0]}")
    print(f"   Output: {output_details[0]}")
    
    # 測試推論
    test_input = np.random.randn(1, freq_bins, time_steps, 1).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"\n   測試推論:")
    print(f"     Input shape: {test_input.shape}")
    print(f"     Output shape: {output.shape}")
    print(f"     Output range: [{output.min():.4f}, {output.max():.4f}]")
    
    # 比較 Keras 和 TFLite 輸出
    print("\n5. 比較 Keras vs TFLite 輸出...")
    keras_output, _ = model(test_input, training=False)
    keras_output = keras_output.numpy()
    
    diff = np.abs(keras_output - output)
    print(f"   最大差異: {diff.max():.6e}")
    print(f"   平均差異: {diff.mean():.6e}")
    
    if diff.max() < 1e-5:
        print("\n🎉 Keras 和 TFLite 輸出幾乎相同！")
    else:
        print("\n⚠️ 輸出有些微差異，可能是數值精度問題")
    
    print("\n" + "=" * 60)
    print("導出完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()

