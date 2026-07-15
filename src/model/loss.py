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


# --loss CLI 인자로 loss를 선택하기 위한 레지스트리.
# 신규 loss 실험을 추가할 때는 클래스를 정의한 뒤 여기에 이름: 클래스 형태로만 등록하면 됨.
LOSS_REGISTRY = {
    'weighted_mse': WeightedMSELossWrapper,
}