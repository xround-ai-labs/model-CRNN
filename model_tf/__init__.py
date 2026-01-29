"""
TensorFlow/Keras 版本的 CRN 模型模組

這些模型是從 PyTorch 版本轉換而來，專為 TFLite 相容性設計。

主要差異：
1. 資料格式：使用 channels_last [B, F, T, C]，而非 PyTorch 的 [B, C, F, T]
2. LSTM：使用 Keras 原生 LSTM，直接支援 TFLite
3. BatchNorm：使用 axis=-1 (channels_last)

使用範例：
    from model_tf import MiniCRN_Causal128
    
    model = MiniCRN_Causal128(n_fft=100)
    
    # 輸入格式 [B, F, T, 1]
    x = tf.random.normal([1, 51, 200, 1])
    y, lstm_states = model(x)
"""

from .crn import (
    CausalConvBlock,
    CausalTransConvBlock,
    MiniCRN_Causal128,
    convert_pytorch_weights_to_keras,
)

from .loss import (
    mse_loss_for_variable_length_data,
    MSELossVariableLength,
)

__all__ = [
    # 模型
    'CausalConvBlock',
    'CausalTransConvBlock',
    'MiniCRN_Causal128',
    # 工具函數
    'convert_pytorch_weights_to_keras',
    # 損失函數
    'mse_loss_for_variable_length_data',
    'MSELossVariableLength',
]

