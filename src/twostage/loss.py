import torch.nn.functional as F
import torch.nn as nn

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

class WeightedMSELossWrapper(nn.Module):
    def __init__(self, alpha=1.5):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, pred, target, mask=None):
        if mask is not None:
            pred = pred[mask]
            target = target[mask]
            
        weight = 1.0 + self.alpha * target
        loss = ((pred - target) ** 2) * weight
        return loss.mean()

