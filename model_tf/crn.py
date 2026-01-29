"""
TensorFlow/Keras 版本的 CRN 模型
從 PyTorch 版本轉換而來，專為 TFLite 相容性設計

注意事項：
1. 使用 channels_last 格式 (TensorFlow 預設): [B, F, T, C]
   - PyTorch 使用 channels_first: [B, C, F, T]
   - 轉換時需要 transpose
2. LSTM 原生支援，不需要透過 ONNX 轉換
3. 所有 shape 在編譯時確定，避免動態形狀問題
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np


class CausalConvBlock(layers.Layer):
    """
    因果卷積區塊 - 確保不會看到未來的資訊
    
    PyTorch 版本:
        Conv2d(in_ch, out_ch, kernel_size=(3,2), stride=(2,1), padding=(0,1))
        然後 x[:, :, :, :-1] 來移除最後一個時間步
    
    Keras 版本:
        先 padding，再 Conv2D，最後裁掉最後一個時間步
    """
    def __init__(self, out_channels, name=None):
        super().__init__(name=name)
        self.out_channels = out_channels
        
    def build(self, input_shape):
        # input_shape: [B, F, T, C]
        # PyTorch padding=(0, 1) 表示時間軸左右各 pad 1
        self.pad = layers.ZeroPadding2D(
            padding=((0, 0), (1, 1))  # (height/freq, width/time) - 時間軸左右各 pad 1
        )
        self.conv = layers.Conv2D(
            filters=self.out_channels,
            kernel_size=(3, 2),
            strides=(2, 1),
            padding='valid',
            data_format='channels_last',
            name=f'{self.name}_conv' if self.name else None
        )
        self.norm = layers.BatchNormalization(
            axis=-1,  # channels_last
            name=f'{self.name}_bn' if self.name else None
        )
        self.activation = layers.ELU()
        super().build(input_shape)
        
    def call(self, x, training=None):
        """
        Args:
            x: [B, F, T, C] - channels_last 格式
        Returns:
            [B, F', T, C'] - F 會因為 stride=2 而減半
        """
        # Pad 時間軸 (左邊 +1)
        x = self.pad(x)
        # 卷積
        x = self.conv(x)
        # 裁掉最後一個時間步 (chomp) - 這是實現因果性的關鍵
        x = x[:, :, :-1, :]
        # 正規化 + 激活
        x = self.norm(x, training=training)
        x = self.activation(x)
        return x


class CausalTransConvBlock(layers.Layer):
    """
    因果轉置卷積區塊 (Decoder 用)
    
    PyTorch 版本:
        ConvTranspose2d(in_ch, out_ch, kernel_size=(3,2), stride=(2,1), output_padding=...)
        然後 x[:, :, :, :-1]
    """
    def __init__(self, out_channels, is_last=False, output_padding=(0, 0), name=None):
        super().__init__(name=name)
        self.out_channels = out_channels
        self.is_last = is_last
        self.output_padding = output_padding
        
    def build(self, input_shape):
        self.conv_transpose = layers.Conv2DTranspose(
            filters=self.out_channels,
            kernel_size=(3, 2),
            strides=(2, 1),
            padding='valid',
            output_padding=self.output_padding,
            data_format='channels_last',
            name=f'{self.name}_convT' if self.name else None
        )
        self.norm = layers.BatchNormalization(
            axis=-1,
            name=f'{self.name}_bn' if self.name else None
        )
        if self.is_last:
            self.activation = layers.ReLU()
        else:
            self.activation = layers.ELU()
        super().build(input_shape)
        
    def call(self, x, training=None):
        """
        Args:
            x: [B, F, T, C]
        Returns:
            [B, F', T, C']
        """
        x = self.conv_transpose(x)
        # Chomp 最後一個時間步
        x = x[:, :, :-1, :]
        x = self.norm(x, training=training)
        x = self.activation(x)
        return x


class MiniCRN_Causal128(Model):
    """
    MiniCRN_Causal128 - 你目前使用的模型
    
    特點：
    - 4 層 encoder
    - 3 層 LSTM (hidden_size=72)
    - 3 層 decoder
    
    Input: [B, F, T, 1] - channels_last 格式
    Output: ([B, F, T, 1], lstm_state)
    
    注意：PyTorch 版本使用 [B, C, F, T]，這裡使用 [B, F, T, C]
    """
    def __init__(self, n_fft=100, name="MiniCRN_Causal128"):
        super().__init__(name=name)
        self.n_fft = n_fft
        self.freq_bins = n_fft // 2 + 1
        
        lstm_size = 72
        self.hidden_size = lstm_size
        self.num_lstm_layers = 3
        
        # Encoder (Causal)
        self.enc1 = CausalConvBlock(16, name='enc1')
        self.enc2 = CausalConvBlock(24, name='enc2')
        self.enc3 = CausalConvBlock(48, name='enc3')
        self.enc4 = CausalConvBlock(lstm_size, name='enc4')
        
        # 計算 encoder 輸出的頻率維度
        # freq_bins=51 經過 4 層 stride=2: 51 -> 26 -> 13 -> 7 -> 4 (取決於 padding)
        # 實際值需要從 PyTorch 模型確認，這裡假設為 3
        self.enc_out_freq = None  # 會在 build 時計算
        
        # LSTM layers (分開定義以支援多層)
        self.lstm_layers = [
            layers.LSTM(
                units=self.hidden_size,
                return_sequences=True,
                return_state=True,

                # 🔴 關鍵參數（一定要加）: 關閉 cuDNN
                implementation=1,   # 使用 non-fused TF graph
                unroll=False,       # 保留動態 time（TFLite 需要）
                use_bias=True,

                name=f'lstm_{i}'
            )
            for i in range(self.num_lstm_layers)
        ]
        
        # Decoder (Causal)
        self.dec1 = CausalTransConvBlock(48, name='dec1')
        self.dec2 = CausalTransConvBlock(24, name='dec2')
        self.dec3 = CausalTransConvBlock(1, is_last=True, name='dec3')
        
    def build(self, input_shape):
        """
        計算 encoder 輸出的頻率維度
        """
        # input_shape: [B, F, T, 1]
        # 模擬 encoder 計算輸出 shape
        f = input_shape[1]
        # 每層 CausalConvBlock 使用 stride=(2,1)，頻率減半
        for _ in range(4):
            # Conv2D with kernel=3, stride=2: f' = floor((f-3+2*pad)/2) + 1
            # 這裡的 padding 是 'valid' + ZeroPadding
            f = (f - 3) // 2 + 1
        self.enc_out_freq = f
        
        # 計算 LSTM input_size
        self.lstm_input_size = self.hidden_size * self.enc_out_freq
        
        super().build(input_shape)
        
    def call(self, x, lstm_states=None, training=None):
        """
        Args:
            x: [B, F, T, 1] - channels_last
            lstm_states: Optional list of (h, c) tuples for each LSTM layer
            training: Boolean for BatchNorm behavior
            
        Returns:
            output: [B, F, T, 1]
            new_lstm_states: List of (h, c) tuples
        """
        # Encoder
        e1 = self.enc1(x, training=training)
        e2 = self.enc2(e1, training=training)
        e3 = self.enc3(e2, training=training)
        e4 = self.enc4(e3, training=training)  # [B, F', T, 72]
        
        # Get shapes
        batch_size = tf.shape(e4)[0]
        freq = e4.shape[1]
        time_steps = tf.shape(e4)[2]
        channels = e4.shape[3]
        
        # Reshape for LSTM: [B, F', T, C] -> [B, T, F'*C]
        lstm_in = tf.transpose(e4, [0, 2, 1, 3])
        lstm_in = tf.reshape(lstm_in, [batch_size, time_steps, freq * channels])
        
        # Multi-layer LSTM
        new_lstm_states = []
        lstm_out = lstm_in
        for i, lstm_layer in enumerate(self.lstm_layers):
            if lstm_states is not None and i < len(lstm_states):
                initial_state = lstm_states[i]
            else:
                initial_state = None
                
            lstm_out, h, c = lstm_layer(
                lstm_out, 
                initial_state=initial_state,
                training=training
            )
            new_lstm_states.append((h, c))
        
        # [B, T, hidden] -> [B, 1, T, hidden]
        lstm_out = tf.reshape(lstm_out, [batch_size, 1, time_steps, self.hidden_size])
        
        # Decoder
        d1 = self.dec1(lstm_out, training=training)
        d2 = self.dec2(d1, training=training)
        out = self.dec3(d2, training=training)

        out = tf.clip_by_value(out, 0.0, 1e6)
        
        # 固定頻率點數
        # out shape: [B, F_out, T, 1]
        f_out = tf.shape(out)[1]

        # pad 到至少 freq_bins
        pad_f = tf.maximum(0, self.freq_bins - f_out)
        out = tf.pad(out, [[0, 0], [0, pad_f], [0, 0], [0, 0]])

        # 再 slice（一定安全）
        out = out[:, :self.freq_bins, :, :]

        
        return out, new_lstm_states
    
    def call_simple(self, x, training=None):
        """
        簡化版 call，不回傳 LSTM state（給 TFLite 用）
        
        Args:
            x: [B, F, T, 1]
        Returns:
            output: [B, F, T, 1]
        """
        out, _ = self.call(x, lstm_states=None, training=training)
        return out


# =====================================================================
# 工具函數：從 PyTorch 權重載入
# =====================================================================

def convert_pytorch_weights_to_keras(pytorch_state_dict, keras_model, n_fft=100):
    """
    將 PyTorch 權重轉換並載入到 Keras 模型
    
    注意：
    - PyTorch Conv2D weight shape: [out_ch, in_ch, H, W]
    - Keras Conv2D weight shape: [H, W, in_ch, out_ch]
    - 需要 transpose
    """
    import numpy as np
    
    # 先 build 模型
    dummy_input = np.zeros((1, n_fft // 2 + 1, 200, 1), dtype=np.float32)
    keras_model(dummy_input, training=False)
    
    # TODO: 實現權重對應邏輯
    # 這需要仔細對應 PyTorch 和 Keras 的層名稱
    
    print("權重轉換功能尚未完整實現，請手動載入或重新訓練")
    return keras_model


# =====================================================================
# 測試程式碼
# =====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("測試 TensorFlow 版本的模型")
    print("=" * 60)
    
    # 測試 MiniCRN_Causal128
    model = MiniCRN_Causal128(n_fft=100)
    
    # 建立測試輸入 (channels_last: [B, F, T, C])
    # 對應 PyTorch 的 [B, C, F, T] = [2, 1, 51, 200]
    x = tf.random.normal([2, 51, 200, 1])
    
    print(f"\nInput shape: {x.shape}")
    
    # 前向傳播
    y, lstm_states = model(x, training=False)
    
    print(f"Output shape: {y.shape}")
    print(f"LSTM states: {len(lstm_states)} layers")
    
    # 顯示模型摘要
    model.summary()
    
    print("\n✅ 模型測試成功！")

