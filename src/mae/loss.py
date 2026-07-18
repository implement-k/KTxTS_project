import torch.nn.functional as F
import torch.nn as nn
import torch

class HuberLossWrapper(nn.Module):
    """
    Pure Huber Loss
    """
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta
    
    def forward(self, pred_log, target_log, mask):
        # Base Loss 
        loss = F.huber_loss(pred_log[mask], target_log[mask], delta=self.delta, reduction='mean')
        return loss


# 이게 정확도가 높게 나옴.
class WeightedMSELossWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target, current_alpha, mask=None):
        if mask is not None:
            pred = pred[mask]
            target = target[mask]
            
        weight = 1.0 + current_alpha * target
        loss = ((pred - target) ** 2) * weight
        return loss.mean()