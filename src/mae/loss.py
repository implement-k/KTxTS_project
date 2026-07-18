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


# 이게 정확도가 높게 나옴.
class WeightedMSELossWrapper(nn.Module):
    def __init__(self, real_penalty_weight=0.005):
        super().__init__()
        self.real_penalty_weight = real_penalty_weight
        
    def forward(self, pred, target, current_alpha, mask=None):
        if mask is not None:
            pred = pred[mask]
            target = target[mask]
            
        # 1. Base Log-scale Weighted MSE (기존: CPC 및 분포 학습용)
        weight = 1.0 + current_alpha * target
        log_loss = ((pred - target) ** 2) * weight
        
        # 2. Real-scale Huber Penalty (추가: 튀는 RMSE 억제용)
        # expm1 연산 시 무한대 발산(Gradient Explosion)을 막기 위해 pred를 12.0 이하로 안전하게 클리핑
        pred_real = torch.expm1(torch.clamp(pred, max=12.0))
        target_real = torch.expm1(target)
        
        # 실제 스케일에서 오차가 100을 넘어가면 선형(L1) 페널티를 주어 튀는 값들을 강하게 억제
        real_loss = F.huber_loss(pred_real, target_real, delta=100.0, reduction='none')
        
        # 최종 Loss: 로그 오차 + (가중치 * 실제 오차)
        loss = log_loss + (self.real_penalty_weight * real_loss)
        return loss.mean()