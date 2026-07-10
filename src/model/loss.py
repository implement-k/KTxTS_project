import torch.nn.functional as F
import torch.nn as nn
import torch

class DynamicPINNLoss(nn.Module):
    """
    Dynamic PINN Loss with Warm-up and Decay Scheduling
    """
    def __init__(self, total_epochs, lambda_start=0.001, lambda_end=0.1, delta_start=1.0, delta_end=0.5):
        super().__init__()
        self.total_epochs = total_epochs
        self.lambda_start = lambda_start
        self.lambda_end = lambda_end
        self.delta_start = delta_start
        self.delta_end = delta_end
    
    def huber_pinn_loss(self, pred_log, target_log, mask, delta=1.0, lambda_marginal=0.1):
        """
        Huber Loss + 물리 제약
        
        - pred_log, target_log: log1p 변환이 적용된 모델 출력값 및 정답 타겟 (Batch, N, N)
        - mask: 손실을 계산할 위치를 나타내는 불리언 텐서
        - delta: Huber Loss의 기준점. 오차가 이 값보다 작으면 L2(MSE), 크면 L1(MAE)으로 작동
        - lambda_marginal: 물리 제약 손실의 반영 비율
        """
        
        # Base Loss 
        # 로그 공간에서의 오차 계산.
        base_loss = F.huber_loss(pred_log[mask], target_log[mask], delta=delta, reduction='mean')
        
        # 물리 스케일 복원
        pred_real = torch.expm1(pred_log)
        target_real = torch.expm1(target_log)
        
        # === 물리적 제약 ===
        # origin total: (Batch, N, N) -> (Batch, N)
        pred_origin_sum = torch.log1p(pred_real.sum(dim=2))
        target_origin_sum = torch.log1p(target_real.sum(dim=2))
        
        # destination total: (Batch, N, N) -> (Batch, N)
        pred_dest_sum = torch.log1p(pred_real.sum(dim=1))
        target_dest_sum = torch.log1p(target_real.sum(dim=1))
        
        # 총량 제약 계산
        origin_loss = F.huber_loss(pred_origin_sum, target_origin_sum, delta=delta*10, reduction='mean')
        dest_loss = F.huber_loss(pred_dest_sum, target_dest_sum, delta=delta*10, reduction='mean')
        
        marginal_loss = origin_loss + dest_loss
        
        # 최종 손실 함수 결합
        total_loss = base_loss + (lambda_marginal * marginal_loss)
        
        return total_loss

    def forward(self, pred_log, target_log, mask, current_epoch):
        progress = min(1.0, current_epoch / max(1, self.total_epochs - 1))
        
        # 후반부로 갈수록 물리 제약 비중 증가
        current_lambda = self.lambda_start + (self.lambda_end - self.lambda_start) * progress
        # 후반부로 갈수록 Huber 임계점을 낮추어 더 깐깐하게 MSE 튜닝 유도
        current_delta = self.delta_start - (self.delta_start - self.delta_end) * progress
        
        total_loss = self.huber_pinn_loss(
            pred_log, target_log, mask,
            delta=current_delta, 
            lambda_marginal=current_lambda
        )
        
        return total_loss
