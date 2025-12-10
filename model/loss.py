import torch
from torch.nn.utils.rnn import pad_sequence

def mse_loss_for_variable_length_data():
    def loss_function(ipt, target, n_frames_list, device):
        """
        Calculate MSE loss for variable-length data.
        ipt:    [B, F, T]
        target: [B, F, T]
        """
        E = 1e-8

        # 🔧 自動裁切時間軸對齊 (避免 1115 vs 1116)
        min_T = min(target.size(-1), ipt.size(-1))
        target = target[..., :min_T]
        ipt = ipt[..., :min_T]

        # 若是 batch=1, 無需 mask
        if target.shape[0] == 1:
            return torch.nn.functional.mse_loss(ipt, target)

        # 建立 mask
        with torch.no_grad():
            masks = []
            for n_frames in n_frames_list:
                masks.append(torch.ones(n_frames, target.size(1), dtype=torch.float32))  # [T_real, F]
            binary_mask = pad_sequence(masks, batch_first=True).to(device).permute(0, 2, 1)  # [B,F,T]

        # 對齊 mask 長度
        min_T = min(binary_mask.size(-1), target.size(-1))
        binary_mask = binary_mask[..., :min_T]
        target = target[..., :min_T]
        ipt = ipt[..., :min_T]

        masked_ipt = ipt * binary_mask
        masked_target = target * binary_mask

        return ((masked_ipt - masked_target) ** 2).sum() / (binary_mask.sum() + E)

    return loss_function
