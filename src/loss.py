import torch.nn.functional as F
import torch.nn as nn
import torch

class HuberODLoss(nn.Module):
    def __init__(self, delta=1.0, tau=0.5):
        super().__init__()
        self.delta = delta
        self.tau = tau
        
    def forward(self, pred, target, alpha, mask):
        if mask.any():
            p = pred[mask]
            t = target[mask]
            
            huber = F.huber_loss(p, t, delta=self.delta, reduction='none')
            
            if self.tau > 0.0:
                weight = torch.pow(t, self.tau)
                loss = (huber * weight).mean()
            else:
                loss = huber.mean()
            return loss
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

class HuberLoss(nn.Module):
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
class WeightedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target, current_alpha, mask=None):
        weight = 1.0 + current_alpha * target
        loss = ((pred - target) ** 2) * weight
        if mask is not None:
            mask_float = mask.float()
            # Calculate mean only over the masked elements to avoid CPU-GPU sync
            loss = (loss * mask_float).sum() / (mask_float.sum() + 1e-8)
            return loss
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


class OffDiagCPCLoss(nn.Module):
    """
    RMSE Collapse 방지용 Loss.

    문제: WeightedMSE는 target이 큰 self-loop(대각선)에 높은 가중치를 줘서
         모델이 off-diagonal을 0으로 예측하는 degenerate solution으로 빠짐.

    해결:
    1. Off-diagonal MSE: 비대각 원소 위주로 loss 계산 (대각선 가중치 대폭 축소)
    2. CPC Loss: 실제 분포의 경향(shape)을 따라가도록 강제
       CPC = 2 * sum(min(p, t)) / (sum(p) + sum(t))  ->  1 - CPC를 최소화

    Args:
        diag_weight: self-loop loss 가중치 (0.1 -> 대각선을 거의 무시)
        cpc_weight:  CPC loss 가중치 (0.5 -> off-diagonal 분포 학습 강제)
    """
    def __init__(self, diag_weight=0.1, cpc_weight=0.5):
        super().__init__()
        self.diag_weight = diag_weight
        self.cpc_weight  = cpc_weight

    def forward(self, pred, target, current_alpha, mask=None):
        B, N, _ = pred.shape
        diag_mask = torch.eye(N, device=pred.device, dtype=torch.bool).unsqueeze(0)  # (1,N,N)

        # 1. Off-diagonal MSE (log scale)
        offdiag_mask = ~diag_mask
        if mask is not None:
            offdiag_mask = offdiag_mask & mask
        if offdiag_mask.sum() > 0:
            loss_offdiag = F.mse_loss(pred[offdiag_mask], target[offdiag_mask])
        else:
            loss_offdiag = pred.new_tensor(0.0)

        # 2. Diagonal MSE (낮은 가중치)
        diag_mask2 = diag_mask
        if mask is not None:
            diag_mask2 = diag_mask & mask
        if diag_mask2.sum() > 0:
            loss_diag = F.mse_loss(pred[diag_mask2], target[diag_mask2])
        else:
            loss_diag = pred.new_tensor(0.0)

        # 3. CPC Loss (real scale)
        # clamp max를 20.0으로 올려 오버플로우만 방지하고 정상적인 gradient 흐름 보장
        pred_real   = torch.expm1(torch.clamp(pred, max=20.0))
        target_real = torch.expm1(target)
        
        # CPC도 **대각선을 제외한(offdiag)** 값들로만 계산해야 self-loop의 영향력을 배제할 수 있음
        if offdiag_mask.sum() > 0:
            p = pred_real[offdiag_mask]
            t = target_real[offdiag_mask]
            cpc = (2.0 * torch.minimum(p, t).sum()) / (p.sum() + t.sum() + 1e-8)
            loss_cpc = 1.0 - cpc
        else:
            loss_cpc = pred.new_tensor(0.0)

        loss = loss_offdiag + self.diag_weight * loss_diag + self.cpc_weight * loss_cpc
        return loss

class CrossEntropyODLoss(nn.Module):
    """
    Cross-Entropy 기반의 분포 학습 Loss.
    
    모델의 출력 pred_od를 logit으로 보고 softmax를 통해 목적지(destination) 분포를 구한 뒤,
    실제 target의 분포와의 Cross Entropy (KL Divergence와 유사)를 계산합니다.
    이 방식은 총량(Volume) 예측보다는 통행이 "어디로" 가는지(Distribution) 패턴을 
    학습하는 데 집중하게 만듭니다.
    """
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred, target, current_alpha=None, mask=None):
        if mask is not None:
            # Mask된 원소만 고려하기 위해 pred의 마스크 밖을 -1e4 처리 (softmax 시 0이 되도록)
            pred = pred.masked_fill(~mask, -1e4)
            target = target * mask.float()

        # origin별(도착지 방향 dim=2)로 softmax 정규화
        log_p_pred = F.log_softmax(pred, dim=2)          
        
        # 실제 타겟의 분포 (합이 1이 되도록)
        # target이 log1p 스케일이므로, 원본 스케일로 복원 후 비율 계산
        target_real = torch.expm1(target)
        p_target = target_real / target_real.sum(dim=2, keepdim=True).clamp(min=1e-8)
        
        # Cross Entropy Loss
        loss_ce = -(p_target * log_p_pred)
        
        if mask is not None:
            loss_ce = loss_ce * mask.float()
            
        return loss_ce.sum(dim=2).mean()

def weibull_nll(pred_lambda, target, k, eps=1e-8):
    """k는 그룹별로 fit한 고정 상수, pred_lambda는 모델이 예측하는 scale parameter"""
    pred_lambda = pred_lambda.clamp(min=eps)
    target = target.clamp(min=eps)
    log_lik = (
        torch.log(torch.tensor(k, device=target.device))
        - k * torch.log(pred_lambda)
        + (k - 1) * torch.log(target)
        - (target / pred_lambda).pow(k)
    )
    return -log_lik.mean()

class HybridWeibullODLoss(nn.Module):
    def __init__(self, k_diag=19.466, k_offdiag=1.213, lambda_diag_weight=0.1, beta=0.5):
        super().__init__()
        self.k_diag = k_diag
        self.k_offdiag = k_offdiag
        self.lambda_diag_weight = lambda_diag_weight
        self.beta = beta
        self.cross_entropy = CrossEntropyODLoss()

    def forward(self, pred_scale, pred_raw, target_od, mask_2d, diag_mask, active_node_mask=None):
        # Scale (weibull) target is original volume. target_od is log1p.
        target_od_real = torch.expm1(target_od)
        
        # target이 0보다 큰(의미 있는 통행이 있는) 곳만 추려내어 valid_mask 업데이트 (weibull 계산 시 0 발산 방지)
        is_positive = target_od_real > 1e-4 
        valid_diag_mask = diag_mask & mask_2d & is_positive
        valid_offdiag_mask = (~diag_mask) & mask_2d & is_positive
        ce_offdiag_mask = (~diag_mask) & mask_2d
        
        if active_node_mask is not None:
            active_node_mask_2d = active_node_mask.unsqueeze(1) & active_node_mask.unsqueeze(2)
            valid_diag_mask = valid_diag_mask & active_node_mask_2d
            valid_offdiag_mask = valid_offdiag_mask & active_node_mask_2d
            ce_offdiag_mask = ce_offdiag_mask & active_node_mask_2d
        
        zero_tensor = torch.tensor(0.0, device=pred_scale.device)
        
        loss_diag = weibull_nll(pred_scale[valid_diag_mask], target_od_real[valid_diag_mask], k=self.k_diag) if valid_diag_mask.any() else zero_tensor
        loss_offdiag = weibull_nll(pred_scale[valid_offdiag_mask], target_od_real[valid_offdiag_mask], k=self.k_offdiag) if valid_offdiag_mask.any() else zero_tensor
        
        loss_scale = loss_offdiag + self.lambda_diag_weight * loss_diag
        
        # Shape loss (Cross Entropy) uses original pred_raw as logits, target_od_real as frequencies
        # Valid offdiag mask for shape doesn't exclude 0 targets, because CE needs to learn the zero shape too
        loss_shape = self.cross_entropy(pred_raw, target_od_real, mask=ce_offdiag_mask)
        
        loss = loss_shape + self.beta * loss_scale
        return loss
