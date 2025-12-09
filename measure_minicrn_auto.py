# measure_model2k7.py
import torch
import torch.nn as nn
from model.crn import MiniCRN

def get_model_size(model: nn.Module):
    """計算模型參數大小 (bytes)"""
    return sum(p.numel() * p.element_size() for p in model.parameters())

def register_activation_hooks(model: nn.Module):
    """註冊 hook，記錄各層 activation 大小"""
    activations = []

    def hook_fn(module, input, output):
        size = 0
        if isinstance(output, (tuple, list)):
            for o in output:
                if isinstance(o, torch.Tensor):
                    size += o.nelement() * o.element_size()
        elif isinstance(output, torch.Tensor):
            size = output.nelement() * output.element_size()
        activations.append(size)

    hooks = []
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.GroupNorm, nn.LSTM, nn.ELU, nn.ReLU)):
            hooks.append(m.register_forward_hook(hook_fn))
    return hooks, activations

def fix_lstm_input_size(model: MiniCRN, input_shape=(2, 1, 161, 200)):
    """根據 Encoder 輸出自動修正 LSTM input_size"""
    x = torch.randn(*input_shape)
    with torch.no_grad():
        e1 = model.act(model.norm1(model.conv1(x)))
        e2 = model.act(model.norm2(model.conv2(e1)))
        e3 = model.act(model.norm3(model.conv3(e2)))
    b, c, f, t = e3.shape
    feat_dim = c * f
    print(f"[Auto-adjust] LSTM input_size → {feat_dim}")
    model.lstm = nn.LSTM(input_size=feat_dim, hidden_size=model.hidden_size, num_layers=1, batch_first=True)

def estimate_forward_memory(model: nn.Module, input_shape=(2, 1, 161, 200)):
    """執行 forward 並統計 activation buffer"""
    x = torch.randn(*input_shape)
    hooks, acts = register_activation_hooks(model)
    with torch.no_grad():
        _ = model(x)
    for h in hooks:
        h.remove()
    total_bytes = sum(acts)
    peak_bytes = max(acts) if acts else 0
    return total_bytes, peak_bytes

if __name__ == "__main__":
    # 你可以在這裡改頻率點數
    input_shape = (2, 1, 51, 200)  # (Batch, Channel, Freq_points, Time_frames)

    model = MiniCRN()
    fix_lstm_input_size(model, input_shape=input_shape)
    model.eval()

    model_bytes = get_model_size(model)
    total_buf, peak_buf = estimate_forward_memory(model, input_shape=input_shape)

    print("=== MiniCRN Memory Report ===")
    print(f"Input shape: {input_shape}")
    print(f"Parameter size: {model_bytes / (1024**2):.3f} MB")
    print(f"Total activations (sum of all): {total_buf / (1024**2):.3f} MB")
    print(f"Peak activation buffer: {peak_buf / (1024**2):.3f} MB")
    print(f"Estimated total (params + peak): {(model_bytes + peak_buf) / (1024**2):.3f} MB")