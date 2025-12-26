import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 2),
            stride=(2, 1),
            padding=(0, 1)
        )
        self.norm = nn.BatchNorm2d(num_features=out_channels)
        self.activation = nn.ELU()

    def forward(self, x):
        """
        2D Causal convolution.
        Args:
            x: [B, C, F, T]

        Returns:
            [B, C, F, T]
        """
        x = self.conv(x)
        x = x[:, :, :, :-1]  # chomp size
        x = self.norm(x)
        x = self.activation(x)
        return x


class CausalTransConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_last=False, output_padding=(0, 0)):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 2),
            stride=(2, 1),
            output_padding=output_padding
        )
        self.norm = nn.BatchNorm2d(num_features=out_channels)
        if is_last:
            self.activation = nn.ReLU()
        else:
            self.activation = nn.ELU()

    def forward(self, x):
        """
        2D Causal convolution.
        Args:
            x: [B, C, F, T]

        Returns:
            [B, C, F, T]
        """
        x = self.conv(x)
        x = x[:, :, :, :-1]  # chomp size
        x = self.norm(x)
        x = self.activation(x)
        return x


class CRN(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """

    def __init__(self):
        super(CRN, self).__init__()
        # Encoder
        self.conv_block_1 = CausalConvBlock(1, 16)
        self.conv_block_2 = CausalConvBlock(16, 32)
        self.conv_block_3 = CausalConvBlock(32, 64)
        self.conv_block_4 = CausalConvBlock(64, 128)
        self.conv_block_5 = CausalConvBlock(128, 256)

        # LSTM
        self.lstm_layer = nn.LSTM(input_size=1024, hidden_size=1024, num_layers=2, batch_first=True)

        self.tran_conv_block_1 = CausalTransConvBlock(256 + 256, 128)
        self.tran_conv_block_2 = CausalTransConvBlock(128 + 128, 64)
        self.tran_conv_block_3 = CausalTransConvBlock(64 + 64, 32)
        self.tran_conv_block_4 = CausalTransConvBlock(32 + 32, 16, output_padding=(1, 0))
        self.tran_conv_block_5 = CausalTransConvBlock(16 + 16, 1, is_last=True)

    def forward(self, x):
        self.lstm_layer.flatten_parameters()

        e_1 = self.conv_block_1(x)
        e_2 = self.conv_block_2(e_1)
        e_3 = self.conv_block_3(e_2)
        e_4 = self.conv_block_4(e_3)
        e_5 = self.conv_block_5(e_4)  # [2, 256, 4, 200]

        batch_size, n_channels, n_f_bins, n_frame_size = e_5.shape

        # [2, 256, 4, 200] = [2, 1024, 200] => [2, 200, 1024]
        lstm_in = e_5.reshape(batch_size, n_channels * n_f_bins, n_frame_size).permute(0, 2, 1)
        lstm_out, _ = self.lstm_layer(lstm_in)  # [2, 200, 1024]
        lstm_out = lstm_out.permute(0, 2, 1).reshape(batch_size, n_channels, n_f_bins, n_frame_size)  # [2, 256, 4, 200]

        d_1 = self.tran_conv_block_1(torch.cat((lstm_out, e_5), 1))
        d_2 = self.tran_conv_block_2(torch.cat((d_1, e_4), 1))
        d_3 = self.tran_conv_block_3(torch.cat((d_2, e_3), 1))
        d_4 = self.tran_conv_block_4(torch.cat((d_3, e_2), 1))
        d_5 = self.tran_conv_block_5(torch.cat((d_4, e_1), 1))

        return d_5

class MiniCRN(nn.Module):
    def __init__(self, n_fft=100):
        super().__init__()

        self.n_fft = n_fft
        self.freq_bins = n_fft // 2 + 1  # 🔸 最終輸出固定頻率點數
        
        # ========== Encoder ==========
        self.conv1 = nn.Conv2d(1, 16, (3,2), stride=(2,1), padding=(1,0))
        self.conv2 = nn.Conv2d(16, 32, (3,2), stride=(2,1), padding=(1,0))
        self.conv3 = nn.Conv2d(32, 64, (3,2), stride=(2,1), padding=(1,0))
        self.norm1 = nn.GroupNorm(1, 16)
        self.norm2 = nn.GroupNorm(1, 32)
        self.norm3 = nn.GroupNorm(1, 64)
        self.act = nn.ELU()

        # ========== LSTM ==========
        self.lstm = None  # 延後初始化
        self.hidden_size = 128  # for reshape

        # ========== Decoder ==========
        self.deconv1 = nn.ConvTranspose2d(128, 64, (3,2), stride=(2,1), output_padding=(1,0))
        self.deconv2 = nn.ConvTranspose2d(64, 32, (3,2), stride=(2,1), output_padding=(1,0))
        self.deconv3 = nn.ConvTranspose2d(32, 16, (3,2), stride=(2,1), output_padding=(1,0))
        self.deconv4 = nn.ConvTranspose2d(16, 1, (3,2), stride=(2,1), output_padding=(1,0))
        
        self.denorm1 = nn.GroupNorm(1, 64)
        self.denorm2 = nn.GroupNorm(1, 32)
        self.denorm3 = nn.GroupNorm(1, 16)

        

    def forward(self, x):
        e1 = self.act(self.norm1(self.conv1(x)))
        e2 = self.act(self.norm2(self.conv2(e1)))
        e3 = self.act(self.norm3(self.conv3(e2)))

        b, c, f, t = e3.shape
        feat_dim = c * f

        # ⚡ LSTM 動態初始化（只第一次執行）
        if self.lstm is None:
            self.lstm = nn.LSTM(input_size=feat_dim, hidden_size=self.hidden_size, num_layers=1, batch_first=True)
            if next(self.parameters()).is_cuda:
                self.lstm = self.lstm.cuda()

        # [B, C, F, T] → [B, T, C×F]
        lstm_in = e3.reshape(b, c*f, t).permute(0, 2, 1)
        lstm_out, _ = self.lstm(lstm_in)  # [B, T, hidden_size]
        lstm_out = lstm_out.permute(0, 2, 1).unsqueeze(2)  # [B, hidden_size, 1, T]
        
        d1 = self.act(self.deconv1(lstm_out))
        d2 = self.act(self.deconv2(d1))
        d3 = self.act(self.deconv3(d2))
        out = torch.relu(self.deconv4(d3))

        # ✅ 固定頻率點數 (n_fft//2 + 1)
        f_out = out.size(2)
        if f_out > self.freq_bins:
            out = out[:, :, :self.freq_bins, :]
        elif f_out < self.freq_bins:
            pad_amt = self.freq_bins - f_out
            out = F.pad(out, (0, 0, 0, pad_amt))  # pad on freq axis (dim=2)
        return out


class MiniCRN_Causal(nn.Module):
    def __init__(self, n_fft=100):
        super().__init__()
        self.freq_bins = n_fft // 2 + 1

        # ===== Encoder (Causal) =====
        self.enc1 = CausalConvBlock(1, 16)
        self.enc2 = CausalConvBlock(16, 32)
        self.enc3 = CausalConvBlock(32, 64)

        # ===== LSTM =====
        self.lstm = None
        self.hidden_size = 64

        # ===== Decoder (Causal) =====
        self.dec1 = CausalTransConvBlock(64, 32)
        self.dec2 = CausalTransConvBlock(32, 16)
        self.dec3 = CausalTransConvBlock(16, 1, is_last=True)

    def forward(self, x):
        """
        x: [B, 1, F, T]
        """
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)   # [B, 64, F', T]

        b, c, f, t = e3.shape
        feat_dim = c * f

        # 動態建立 LSTM（跟你原本 MiniCRN 一樣）
        if self.lstm is None:
            self.lstm = nn.LSTM(
                input_size=feat_dim,
                hidden_size=self.hidden_size,
                num_layers=1,
                batch_first=True
            )
            if x.is_cuda:
                self.lstm = self.lstm.cuda()

        lstm_in = e3.reshape(b, feat_dim, t).permute(0, 2, 1)
        lstm_out, _ = self.lstm(lstm_in)
        lstm_out = lstm_out.permute(0, 2, 1).reshape(b, self.hidden_size, 1, t)

        d1 = self.dec1(lstm_out)
        d2 = self.dec2(d1)
        out = self.dec3(d2)

        # 固定頻率點數
        f_out = out.size(2)
        if f_out > self.freq_bins:
            out = out[:, :, :self.freq_bins, :]
        elif f_out < self.freq_bins:
            out = F.pad(out, (0, 0, 0, self.freq_bins - f_out))

        return out

class MiniCRN_Causal128(nn.Module):
    def __init__(self, n_fft=100):
        super().__init__()
        self.freq_bins = n_fft // 2 + 1

        lstm_size = 72

        # ===== Encoder (Causal) =====
        self.enc1 = CausalConvBlock(1, 16)
        self.enc2 = CausalConvBlock(16, 24)
        self.enc3 = CausalConvBlock(24, 48)
        self.enc4 = CausalConvBlock(48, lstm_size)

        # ===== LSTM =====
        self.hidden_size = lstm_size
        self.num_lstm_layers = 3

        # ===== Auto infer encoder output freq =====
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.freq_bins, 8)
            e = self.enc4(self.enc3(self.enc2(self.enc1(dummy))))
            self.enc_out_freq = e.shape[2]   # ← 自動推斷

        # 固定 input_size
        feat_dim = lstm_size * self.enc_out_freq

        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_lstm_layers,
            batch_first=True
        )

        # ===== Decoder (Causal) =====
        self.dec1 = CausalTransConvBlock(lstm_size, 48)   # 注意：Decoder 第一層 input 要接 LSTM hidden_size
        self.dec2 = CausalTransConvBlock(48, 24)
        self.dec3 = CausalTransConvBlock(24, 1, is_last=True)

    def forward(self, x):
        """
        x: [B, 1, F, T]
        """
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3) # [B, 64, F', T]

        b, c, f, t = e4.shape

        # ===== 保護性檢查 =====
        assert f == self.enc_out_freq, \
            f"Encoder freq mismatch: expect {self.enc_out_freq}, got {f}"

        # [B, C, F, T] → [B, T, C*F]
        lstm_in = e4.reshape(b, c * f, t).permute(0, 2, 1)

        lstm_out, _ = self.lstm(lstm_in)

        lstm_out = torch.nan_to_num(
            lstm_out,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        # [B, T, 128] → [B, 128, 1, T]
        lstm_out = lstm_out.permute(0, 2, 1).unsqueeze(2)

        d1 = self.dec1(lstm_out)
        d2 = self.dec2(d1)
        out = self.dec3(d2)

        out = torch.nan_to_num(
            out,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        # ===== Fix frequency bins =====
        f_out = out.size(2)
        if f_out > self.freq_bins:
            out = out[:, :, :self.freq_bins, :]
        elif f_out < self.freq_bins:
            out = F.pad(
                out,
                (0, 0, 0, self.freq_bins - f_out),
                mode="constant",
                value=0.0,   # 明確指定
            )

        return out

if __name__ == "__main__":
    model = MiniCRN_Causal128(n_fft=100)
    x = torch.randn(2, 1, 51, 200)
    y = model(x)
    print("input:", x.shape)
    print("output:", y.shape)

    with torch.no_grad():
        x = torch.randn(2, 1, 51, 200)
        e4 = model.enc4(model.enc3(model.enc2(model.enc1(x))))
        print(e4.shape)  # 看 F'

    if model.lstm is not None:
        print("LSTM layers :", model.lstm.num_layers)
        print("LSTM hidden :", model.lstm.hidden_size)
        print("LSTM input  :", model.lstm.input_size)

