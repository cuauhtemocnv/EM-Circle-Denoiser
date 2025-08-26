# utils.py
import torch
import torch.nn.functional as F

def weighted_mse(output, target, weight=10.0):
    """Weighted MSE, foreground pixels weighted more."""
    fg = target > 0.1
    loss_fg = F.mse_loss(output[fg], target[fg]) if fg.any() else 0
    loss_bg = F.mse_loss(output[~fg], target[~fg])
    return loss_bg + weight * loss_fg
