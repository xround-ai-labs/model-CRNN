import torch
import torch.nn as nn

# 請確認這裡 import 的就是你目前在用的模型
from model.crn import MiniCRN_Causal128


# ==========================
# Parameter size
# ==========================
def get_model_size(model: nn.Module):
    """計算模型參數大小 (bytes)"""
    return sum(p.numel() * p.element_size() for p in model.parameters())


# ==========================
# Activation hooks
# ==========================
def register_activation_hooks(model: nn.Module):
    """
    註冊 forward hook，統計每一層 activation 的 byte size
    """
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
        if isinstance(
            m,
            (
                nn.Conv2d,
                nn.ConvTranspose2d,
                nn.GroupNorm,
                nn.LSTM,
                nn.ELU,
                nn.ReLU,
            ),
        ):
            hooks.append(m.register_forward_hook(hook_fn))

    return hooks, activations


# ==========================
# Forward memory estimation
# ==========================
def estimate_forward_memory(model: nn.Module, input_shape):
    """
    執行一次 forward，量測 activation buffer
    """
    x = torch.randn(*input_shape)

    hooks, acts = register_activation_hooks(model)
    with torch.no_grad():
        _ = model(x)

    for h in hooks:
        h.remove()

    total_bytes = sum(acts)
    peak_bytes = max(acts) if acts else 0
    return total_bytes, peak_bytes


# ==========================
# Main
# ==========================
if __name__ == "__main__":
    # ===== 可自行調整 =====
    input_shape = (2, 1, 51, 200)  # (B, C, F, T)
    n_fft = 100

    # ===== Model =====
    model = MiniCRN_Causal128(n_fft=n_fft)
    model.eval()

    # ===== Measure =====
    model_bytes = get_model_size(model)
    total_buf, peak_buf = estimate_forward_memory(model, input_shape)

    print("=== MiniCRN_Causal Memory Report ===")
    print(f"Input shape: {input_shape}")
    print(f"Parameter size: {model_bytes / (1024**2):.3f} MB")
    print(f"Total activations (sum): {total_buf / (1024**2):.3f} MB")
    print(f"Peak activation buffer: {peak_buf / (1024**2):.3f} MB")
    print(f"Estimated total (params + peak): {(model_bytes + peak_buf) / (1024**2):.3f} MB")
