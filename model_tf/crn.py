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
            axis=-1,
            momentum=0.9,        # 🔴 降低動量，讓統計量更新更穩
            epsilon=1e-5,        # 🔴 提高數值穩定性
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
            momentum=0.9,        # 🔴 降低動量，讓統計量更新更穩
            epsilon=1e-5,        # 🔴 提高數值穩定性
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
    MiniCRN_Causal128 - full-band / encoder=3 版本（TFLite 固定輸入 51x200 仍可用）

    重點修正：
    - Encoder 改為 3 層（stride=(2,1)）：freq 51 -> 25 -> 12 -> 5
    - Bottleneck 不再把 freq 硬壓成 1；改為保留 enc_out_freq (=5)，避免高頻永遠被 pad 成 0
    - Decoder 仍為 3 層，但使用 output_padding=(1,0) 讓 freq: 5 -> 12 -> 25 -> 52，再 slice 成 51
    - 參數量控制在 <1MB（預設 hidden_size=64, num_lstm_layers=2）

    Input:  [B, F=51, T=200, 1] (channels_last)
    Output: ([B, 51, 200, 1])

    注意：此版本架構與舊版不相容，需重新訓練與重新匯出 TFLite。
    """

    def __init__(
        self,
        n_fft: int = 100,
        hidden_size: int = 64,
        num_lstm_layers: int = 2,
        name: str = "MiniCRN_Causal128",
    ):
        super().__init__(name=name)
        self.n_fft = n_fft
        self.freq_bins = n_fft // 2 + 1  # 51 for n_fft=100

        self.hidden_size = int(hidden_size)
        self.num_lstm_layers = int(num_lstm_layers)

        # -------------------------
        # Encoder (Causal)
        # -------------------------
        self.enc1 = CausalConvBlock(16, name='enc1')
        self.enc2 = CausalConvBlock(24, name='enc2')
        self.enc3 = CausalConvBlock(48, name='enc3')

        self.enc_out_freq = None  # set in build()

        # -------------------------
        # LSTM stack (stateful-friendly)
        # -------------------------
        self.lstm_layers = [
            layers.LSTM(
                units=self.hidden_size,
                return_sequences=True,
                return_state=False,

                # non-fused graph to be TFLite-friendly
                implementation=1,
                unroll=False,
                use_bias=True,
                name=f'lstm_{i}'
            )
            for i in range(self.num_lstm_layers)
        ]

        # Will be created in build() after enc_out_freq known
        self.proj = None

        # -------------------------
        # Decoder (Causal)
        # Use output_padding=(1,0) to make freq upscale hit 52 then slice->51
        # -------------------------
        self.dec1 = CausalTransConvBlock(48, output_padding=(1, 0), name='dec1')
        self.dec2 = CausalTransConvBlock(24, output_padding=(1, 0), name='dec2')
        self.dec3 = CausalTransConvBlock(1, is_last=True, output_padding=(1, 0), name='dec3')

    def build(self, input_shape):
        """
        計算 encoder 輸出的頻率維度並建立 projection。
        """
        # input_shape: [B, F, T, 1]
        f = int(input_shape[1])

        # Encoder is 3 layers; each CausalConvBlock uses kernel_size=(3,2), stride=(2,1), no freq padding.
        # freq_out = floor((f - 3)/2) + 1
        for _ in range(3):
            f = (f - 3) // 2 + 1

        self.enc_out_freq = int(f)  # expected 5 when F=51

        # Projection: [B,T,H] -> [B,T,enc_out_freq*H] so we can reshape back to [B,enc_out_freq,T,H]
        self.proj = layers.Dense(
            units=self.enc_out_freq * self.hidden_size,
            use_bias=True,
            name="lstm_to_dec_proj"
        )

        super().build(input_shape)

    def call(self, x, training=None):
        """
        Args:
            x: [B, F, T, 1]
            lstm_states: Optional list of (h, c) tuples for each LSTM layer
            training: Boolean for BatchNorm behavior

        Returns:
            out: [B, F, T, 1]
        """
        # -------------------------
        # Encoder
        # -------------------------
        e1 = self.enc1(x, training=training)
        e2 = self.enc2(e1, training=training)
        e3 = self.enc3(e2, training=training)  # [B, F'=enc_out_freq, T, C=48]

        batch_size = tf.shape(e3)[0]
        time_steps = tf.shape(e3)[2]
        freq = tf.shape(e3)[1]
        ch = e3.shape[3]

        # [B, F', T, C] -> [B, T, F'*C]
        lstm_in = tf.transpose(e3, [0, 2, 1, 3])
        lstm_in = tf.reshape(lstm_in, [batch_size, time_steps, freq * ch])

        # -------------------------
        # LSTM stack
        # -------------------------
        lstm_out = lstm_in
        for lstm_layer in self.lstm_layers:
            lstm_out = lstm_layer(lstm_out, training=training)

        # -------------------------
        # Restore freq dimension for decoder: keep enc_out_freq (NOT 1)
        # lstm_out: [B, T, H]
        # proj -> [B, T, F'*H] -> reshape -> [B, F', T, H]
        # -------------------------
        proj = self.proj(lstm_out)  # [B, T, F'*H]
        proj = tf.reshape(
            proj,
            [batch_size, time_steps, self.enc_out_freq, self.hidden_size]
        )
        dec_in = tf.transpose(proj, [0, 2, 1, 3])  # [B,F',T,H]

        # -------------------------
        # Decoder
        # -------------------------
        d1 = self.dec1(dec_in, training=training)
        d2 = self.dec2(d1, training=training)
        out = self.dec3(d2, training=training)

        # Safety clamp (avoid NaNs/inf propagation)
        out = tf.clip_by_value(out, 0.0, 1e6)

        # Ensure exact freq_bins (51): slice after pad if needed
        f_out = tf.shape(out)[1]
        pad_f = tf.maximum(0, self.freq_bins - f_out)
        out = tf.pad(out, [[0, 0], [0, pad_f], [0, 0], [0, 0]])
        out = out[:, :self.freq_bins, :, :]

        return out

    def call_simple(self, x, training=None):
        """
        簡化版 call，不回傳 LSTM state（給 TFLite 用）
        """
        return self.call(x, training=training)


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

