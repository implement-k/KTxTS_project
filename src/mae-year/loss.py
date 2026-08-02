import torch.nn.functional as F
import torch.nn as nn
import torch

class HuberLoss(nn.Module):
    """
    Pure Huber Loss
    """
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta
    
    def forward(self, pred_log, target_log, current_alpha=None, mask=None):
        # current_alpha is accepted for the shared train-loop call contract and
        # intentionally ignored by plain Huber.
        if mask is None and isinstance(current_alpha, torch.Tensor):
            mask = current_alpha
        del current_alpha
        if mask is not None:
            pred_log = pred_log[mask]
            target_log = target_log[mask]
        if pred_log.numel() == 0:
            return pred_log.sum() * 0.0
        loss = F.huber_loss(pred_log, target_log, delta=self.delta, reduction='mean')
        return loss


# 이게 정확도가 높게 나옴.
class WeightedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target, current_alpha, mask=None):
        if mask is not None:
            pred = pred[mask]
            target = target[mask]
            
        weight = 1.0 + current_alpha * target
        loss = ((pred - target) ** 2) * weight
        return loss.mean()

class HybridWeightedMSELoss(nn.Module):
    def __init__(self, real_penalty_weight=0.005):
        super().__init__()
        self.real_penalty_weight = real_penalty_weight
        
    def forward(self, pred, target, current_alpha, mask=None):
        if mask is not None:
            pred = pred[mask]
            target = target[mask]
            
        # 1. Base Log-scale Weighted MSE
        weight = 1.0 + current_alpha * target
        log_loss = ((pred - target) ** 2) * weight
        
        # 2. Real-scale Huber Penalty
        pred_real = torch.expm1(torch.clamp(pred, max=12.0))
        target_real = torch.expm1(target)
        
        real_loss = F.huber_loss(pred_real, target_real, delta=100.0, reduction='none')
        
        loss = log_loss + (self.real_penalty_weight * real_loss)
        return loss.mean()
