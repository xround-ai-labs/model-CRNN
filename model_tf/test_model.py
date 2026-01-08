"""
測試 TensorFlow 版本的模型

執行方式:
    cd /home/user/Documents/ai-training/model_survey/model-CRNN
    python -m model_tf.test_model
"""

import os
import sys

# 確保可以 import model_tf
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf

from model_tf import (
    CausalConvBlock,
    CausalTransConvBlock,
    MiniCRN_Causal128,
    mse_loss_for_variable_length_data,
)


def test_causal_conv_block():
    """測試 CausalConvBlock"""
    print("\n" + "=" * 60)
    print("測試 CausalConvBlock")
    print("=" * 60)
    
    # [B, F, T, C]
    x = tf.random.normal([2, 51, 200, 1])
    print(f"Input shape: {x.shape}")
    
    block = CausalConvBlock(16, name='test_conv')
    y = block(x, training=False)
    
    print(f"Output shape: {y.shape}")
    print("✅ CausalConvBlock 測試通過")
    return True


def test_causal_trans_conv_block():
    """測試 CausalTransConvBlock"""
    print("\n" + "=" * 60)
    print("測試 CausalTransConvBlock")
    print("=" * 60)
    
    # 假設經過多層 encoder 後的 shape
    x = tf.random.normal([2, 1, 200, 72])
    print(f"Input shape: {x.shape}")
    
    block = CausalTransConvBlock(48, name='test_trans_conv')
    y = block(x, training=False)
    
    print(f"Output shape: {y.shape}")
    print("✅ CausalTransConvBlock 測試通過")
    return True


def test_minicrn_causal128():
    """測試 MiniCRN_Causal128"""
    print("\n" + "=" * 60)
    print("測試 MiniCRN_Causal128")
    print("=" * 60)
    
    model = MiniCRN_Causal128(n_fft=100)
    
    # 對應 PyTorch [2, 1, 51, 200] -> TF [2, 51, 200, 1]
    x = tf.random.normal([2, 51, 200, 1])
    print(f"Input shape: {x.shape}")
    
    # 前向傳播
    y, lstm_states = model(x, training=False)
    
    print(f"Output shape: {y.shape}")
    print(f"Expected shape: [2, 51, 200, 1]")
    print(f"LSTM states: {len(lstm_states)} layers")
    for i, (h, c) in enumerate(lstm_states):
        print(f"  Layer {i}: h={h.shape}, c={c.shape}")
    
    # 驗證輸出 shape
    assert y.shape[0] == 2, "Batch size mismatch"
    assert y.shape[1] == 51, "Freq bins mismatch"
    assert y.shape[2] == 200, "Time steps mismatch"
    assert y.shape[3] == 1, "Channels mismatch"
    
    print("✅ MiniCRN_Causal128 測試通過")
    return True


def test_minicrn_causal128_streaming():
    """測試 MiniCRN_Causal128 串流模式（帶 LSTM state）"""
    print("\n" + "=" * 60)
    print("測試 MiniCRN_Causal128 串流模式")
    print("=" * 60)
    
    model = MiniCRN_Causal128(n_fft=100)
    
    # 第一個 chunk
    x1 = tf.random.normal([1, 51, 10, 1])
    print(f"Chunk 1 shape: {x1.shape}")
    y1, states = model(x1, training=False)
    print(f"Output 1 shape: {y1.shape}")
    
    # 第二個 chunk，使用之前的 state
    x2 = tf.random.normal([1, 51, 10, 1])
    print(f"Chunk 2 shape: {x2.shape}")
    y2, states = model(x2, lstm_states=states, training=False)
    print(f"Output 2 shape: {y2.shape}")
    
    print("✅ 串流模式測試通過")
    return True


def test_loss_function():
    """測試損失函數"""
    print("\n" + "=" * 60)
    print("測試損失函數")
    print("=" * 60)
    
    loss_fn = mse_loss_for_variable_length_data()
    
    # 測試資料
    pred = tf.random.normal([4, 51, 200])
    target = tf.random.normal([4, 51, 200])
    n_frames = [200, 180, 150, 100]
    
    # 無 mask
    loss_no_mask = loss_fn(pred, target, n_frames_list=None)
    print(f"Loss (no mask): {loss_no_mask:.6f}")
    
    # 有 mask
    loss_with_mask = loss_fn(pred, target, n_frames_list=n_frames)
    print(f"Loss (with mask): {loss_with_mask:.6f}")
    
    print("✅ 損失函數測試通過")
    return True


def test_model_summary():
    """顯示模型摘要"""
    print("\n" + "=" * 60)
    print("模型摘要")
    print("=" * 60)
    
    model = MiniCRN_Causal128(n_fft=100)
    
    # Build 模型
    x = tf.random.normal([1, 51, 200, 1])
    model(x, training=False)
    
    model.summary()
    
    # 計算參數量
    total_params = model.count_params()
    print(f"\n總參數量: {total_params:,}")
    print(f"約 {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    return True


def test_tflite_conversion():
    """測試 TFLite 轉換"""
    print("\n" + "=" * 60)
    print("測試 TFLite 轉換")
    print("=" * 60)
    
    model = MiniCRN_Causal128(n_fft=100)
    
    # Build 模型
    x = tf.random.normal([1, 51, 200, 1])
    model(x, training=False)
    
    # 建立用於 TFLite 的 Concrete Function
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[1, 51, 200, 1], dtype=tf.float32)
    ])
    def model_fn(x):
        out, _ = model(x, training=False)
        return out
    
    # 轉換為 TFLite
    concrete_func = model_fn.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    
    # 設定轉換選項
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
    ]
    
    try:
        tflite_model = converter.convert()
        
        # 儲存測試用的 TFLite 模型
        test_tflite_path = "/tmp/test_minicrn_causal128.tflite"
        with open(test_tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"TFLite 模型大小: {len(tflite_model) / 1024:.2f} KB")
        print(f"測試模型儲存於: {test_tflite_path}")
        
        # 測試 TFLite 推論
        interpreter = tf.lite.Interpreter(model_path=test_tflite_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"\nTFLite Input: {input_details[0]['shape']}")
        print(f"TFLite Output: {output_details[0]['shape']}")
        
        # 執行推論
        test_input = np.random.randn(1, 51, 200, 1).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        
        print(f"TFLite 推論輸出 shape: {output.shape}")
        print("✅ TFLite 轉換測試通過！")
        
        return True
        
    except Exception as e:
        print(f"❌ TFLite 轉換失敗: {e}")
        return False


def main():
    """執行所有測試"""
    print("=" * 60)
    print("TensorFlow 模型測試")
    print("=" * 60)
    print(f"TensorFlow 版本: {tf.__version__}")
    
    tests = [
        ("CausalConvBlock", test_causal_conv_block),
        ("CausalTransConvBlock", test_causal_trans_conv_block),
        ("MiniCRN_Causal128", test_minicrn_causal128),
        ("串流模式", test_minicrn_causal128_streaming),
        ("損失函數", test_loss_function),
        ("模型摘要", test_model_summary),
        ("TFLite 轉換", test_tflite_conversion),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success))
        except Exception as e:
            print(f"❌ {name} 測試失敗: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # 顯示結果摘要
    print("\n" + "=" * 60)
    print("測試結果摘要")
    print("=" * 60)
    
    for name, success in results:
        status = "✅ 通過" if success else "❌ 失敗"
        print(f"  {name}: {status}")
    
    all_passed = all(success for _, success in results)
    if all_passed:
        print("\n🎉 所有測試通過！")
    else:
        print("\n⚠️ 部分測試失敗，請檢查上方錯誤訊息")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

