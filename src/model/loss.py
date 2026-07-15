import torch
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


class DualScaleMSELoss(nn.Module):
    """
    log1p 공간 MSE(L_log) + real-scale MSE(L_real, detached scale로 정규화)를 결합.

    L_log  = MSE(pred_log, target_log)
    L_real = MSE(clamp(expm1(pred_log), min=0), expm1(target_log)) / detached_scale
    L      = lambda_log * L_log + (1 - lambda_log) * L_real

    detached_scale: target_real의 robust한 크기(RMS) 기준. global_scale을 생성자에 넘기면
    (target_distribution 분석에서 미리 계산한 학습 데이터 전역 통계) 그 값을 고정으로 사용하고,
    없으면 배치별 detached RMS로 폴백한다(둘 다 gradient가 흐르지 않도록 detach).
    pred_log를 expm1 이전에 clip해 수치 폭발(inf)을 방지한다.
    """
    def __init__(self, lambda_log=0.5, global_scale=None, eps=1e-2, min_scale=1.0, pred_log_clip=20.0):
        super().__init__()
        self.lambda_log = lambda_log
        self.eps = eps
        self.min_scale = min_scale
        self.pred_log_clip = pred_log_clip
        gs = float(global_scale) if global_scale is not None else -1.0
        self.register_buffer('global_scale', torch.tensor(gs))

    def forward(self, pred_log, target_log, mask=None):
        if mask is not None:
            pred_log = pred_log[mask]
            target_log = target_log[mask]

        L_log = F.mse_loss(pred_log, target_log, reduction='mean')

        pred_log_clamped = torch.clamp(pred_log, max=self.pred_log_clip)
        pred_real = torch.clamp(torch.expm1(pred_log_clamped), min=0.0)
        target_real = torch.expm1(target_log)

        if self.global_scale.item() > 0:
            scale = self.global_scale
        else:
            scale = torch.sqrt(torch.mean(target_real.detach() ** 2) + self.eps)
        scale = torch.clamp(scale, min=self.min_scale).detach()

        L_real = F.mse_loss(pred_real, target_real, reduction='mean') / (scale + self.eps)

        return self.lambda_log * L_log + (1.0 - self.lambda_log) * L_real


class BinBalancedMSELoss(nn.Module):
    """
    통행량 크기 구간(0 / 1~100 / 101~999 / 1000이상)별 inverse-frequency 가중치를 적용한
    log1p 공간 MSE. 표본이 압도적으로 많은 소형/동일동 OD가 loss를 독점하지 않도록 함.

    - 구간별 전역 표본 비율(bin_freq)은 target_distribution 분석 결과를 생성자에 주입해서 사용
      (없으면 균등 분포로 폴백하되, 이는 사실상 가중치를 걸지 않는 것과 같아 권장하지 않음).
    - weight_mode='inv_freq' 또는 'inv_sqrt_freq' 중 선택(제곱근 버전이 더 완만한 가중치).
    - 전역 가중치를 계산한 뒤 최대값을 clamp하고, 배치 단위로 다시 평균 1로 정규화한 후
      한 번 더 clamp해 특정 배치 구성에서 가중치가 과도해지는 것을 막는다.
    - 가중치 계산에 쓰는 target은 반드시 detach(그래디언트가 가중치를 통해 흐르지 않게 함).
    - 동일동/동간 여부는 여기서 쓰지 않음(평가에서만 구분).
    """
    def __init__(self, bin_freq=None, weight_mode='inv_sqrt_freq', max_weight=10.0, eps=1e-6):
        super().__init__()
        if bin_freq is None:
            bin_freq = [0.25, 0.25, 0.25, 0.25]
        bin_freq_t = torch.clamp(torch.tensor(bin_freq, dtype=torch.float32), min=eps)

        if weight_mode == 'inv_freq':
            raw_w = 1.0 / bin_freq_t
        elif weight_mode == 'inv_sqrt_freq':
            raw_w = 1.0 / torch.sqrt(bin_freq_t)
        else:
            raise ValueError(f"알 수 없는 weight_mode: {weight_mode}")

        raw_w = raw_w / raw_w.mean()
        raw_w = torch.clamp(raw_w, max=max_weight)

        self.register_buffer('bin_weights', raw_w)  # (4,) : [0, 1~100, 101~999, 1000+]
        self.max_weight = max_weight

    @staticmethod
    def _bin_index(target_real):
        # 0 / (0,100] / (100,1000) / [1000,∞) — 겹침·빈틈 없는 완전 분할.
        # OD 값이 소수로도 존재하므로 정수 경계(>=1, >=101 등)를 쓰면 (0,1)/(100,101)/(999,1000)
        # 구간의 값이 어느 bin에도 속하지 못하는 문제가 있어 이렇게 정의함(src/eval_utils.py와 동일).
        idx = torch.zeros_like(target_real, dtype=torch.long)
        idx = torch.where((target_real > 0) & (target_real <= 100), torch.ones_like(idx), idx)
        idx = torch.where((target_real > 100) & (target_real < 1000), torch.full_like(idx, 2), idx)
        idx = torch.where(target_real >= 1000, torch.full_like(idx, 3), idx)
        return idx

    def forward(self, pred_log, target_log, mask=None):
        if mask is not None:
            pred_log = pred_log[mask]
            target_log = target_log[mask]

        target_real = torch.expm1(target_log).detach()
        bin_idx = self._bin_index(target_real)
        weight = self.bin_weights.to(pred_log.device)[bin_idx]

        batch_mean = weight.mean().clamp(min=1e-6)
        weight = torch.clamp(weight / batch_mean, max=self.max_weight)

        se = (pred_log - target_log) ** 2
        return (se * weight).mean()


class CPCHybridLoss(nn.Module):
    """
    log1p 공간 MSE + (1 - SoftCPC)를 결합. 최종 공식 평가 지표인 CPC를 직접 겨냥.

    SoftCPC = 2 * sum(soft_min(pred_real, target_real)) / (sum(pred_real + target_real) + eps)
    soft_min(a, b) = (a + b - |a - b|) / 2   [min(a,b)의 정확한 항등식, 온도 파라미터 없이
        어디서나(subgradient 포함) 미분 가능 — 온도 기반 softmin 근사보다 수치적으로 더 안정적]
    L = MSE(pred_log, target_log) + lambda_cpc * (1 - SoftCPC)

    퇴행적 해 확인: pred_real을 전체적으로 키우면(>= target_real인 영역) soft_min이
    target_real로 고정되는 반면 분모(pred+target 합)는 커지므로 SoftCPC는 오히려 감소한다.
    즉 "예측 총량을 무조건 키워 CPC를 속이는" 해는 이 식에서 이득이 되지 않으며,
    L_log_mse 항이 추가로 이를 억제한다.
    """
    def __init__(self, lambda_cpc=0.1, eps=1e-6, pred_log_clip=20.0):
        super().__init__()
        self.lambda_cpc = lambda_cpc
        self.eps = eps
        self.pred_log_clip = pred_log_clip

    def forward(self, pred_log, target_log, mask=None):
        if mask is not None:
            pred_log = pred_log[mask]
            target_log = target_log[mask]

        L_log_mse = F.mse_loss(pred_log, target_log, reduction='mean')

        pred_log_clamped = torch.clamp(pred_log, max=self.pred_log_clip)
        pred_real = torch.clamp(torch.expm1(pred_log_clamped), min=0.0)
        target_real = torch.expm1(target_log)

        soft_min = (pred_real + target_real - torch.abs(pred_real - target_real)) / 2.0
        numerator = 2.0 * soft_min.sum()
        denominator = (pred_real + target_real).sum() + self.eps
        soft_cpc = numerator / denominator

        return L_log_mse + self.lambda_cpc * (1.0 - soft_cpc)


class TweedieDevianceLoss(nn.Module):
    """
    Tweedie deviance loss (1 < p < 2). 0이 많고 우측 꼬리가 두꺼운 카운트성 OD 데이터에 적합.

    d(y, mu) = 2 * ( y^(2-p)/((1-p)(2-p)) - y*mu^(1-p)/(1-p) + mu^(2-p)/(2-p) )

    - y = expm1(target_log) (항상 0 이상)
    - mu = clamp(expm1(clamp(pred_log, max=pred_log_clip)), min=eps) : mu는 반드시 양수여야
      하므로 clamp(min=eps)로 확보한다.
    - 모델 출력층/구조(embedding, Transformer 등)는 변경하지 않는다: 위 clamp만으로 mu>0이
      보장되므로 별도 Softplus 출력층 추가는 이번 1차 실험에서 불필요하다고 판단해 제외하지
      않고 그대로 구현함(제외 사유 기록 요구사항에 대한 결론: "제외 안 함, clamp로 충분").
    """
    def __init__(self, p=1.5, eps=1e-3, pred_log_clip=20.0):
        super().__init__()
        assert 1.0 < p < 2.0, "Tweedie power p는 (1, 2) 범위여야 함"
        self.p = p
        self.eps = eps
        self.pred_log_clip = pred_log_clip

    def forward(self, pred_log, target_log, mask=None):
        if mask is not None:
            pred_log = pred_log[mask]
            target_log = target_log[mask]

        p = self.p
        y = torch.clamp(torch.expm1(target_log), min=0.0)
        pred_log_clamped = torch.clamp(pred_log, max=self.pred_log_clip)
        mu = torch.clamp(torch.expm1(pred_log_clamped), min=self.eps)

        term1 = torch.pow(y, 2 - p) / ((1 - p) * (2 - p))
        term2 = y * torch.pow(mu, 1 - p) / (1 - p)
        term3 = torch.pow(mu, 2 - p) / (2 - p)

        deviance = 2.0 * (term1 - term2 + term3)
        return deviance.mean()


class TailAwareRelativeLoss(nn.Module):
    """
    작은 값에는 log1p-MSE(사실상 절대오차 성격)를, 큰 값에는 상대오차를 추가로 반영.

    relative_error = (pred_real - target_real) / (target_real + tau)   [target_real > 0 인 항목만]
    L = lambda_log * MSE(pred_log, target_log) + lambda_relative * mean(relative_error^2)

    - MAPE(평균절대백분율오차)를 그대로 쓰지 않고, tau로 분모를 이동시킨 완화된 상대오차의
      "제곱평균"을 사용한다.
    - target_real == 0인 항목은 상대오차 항 계산에서 제외한다(0 근처 폭발 방지, 대신 L_log가
      이 항목들의 학습을 담당).
    - tau는 생성자 인자로 주입(target_distribution 분석에서 계산한 양수 OD의 median 등을
      사용하는 것을 권장, 기본값은 데이터 미주입 시의 안전한 폴백일 뿐).
    """
    def __init__(self, lambda_log=0.7, lambda_relative=0.3, tau=50.0, pred_log_clip=20.0):
        super().__init__()
        self.lambda_log = lambda_log
        self.lambda_relative = lambda_relative
        self.pred_log_clip = pred_log_clip
        self.register_buffer('tau', torch.tensor(float(tau)))

    def forward(self, pred_log, target_log, mask=None):
        if mask is not None:
            pred_log = pred_log[mask]
            target_log = target_log[mask]

        L_log = F.mse_loss(pred_log, target_log, reduction='mean')

        pred_log_clamped = torch.clamp(pred_log, max=self.pred_log_clip)
        pred_real = torch.clamp(torch.expm1(pred_log_clamped), min=0.0)
        target_real = torch.expm1(target_log)

        nonzero_mask = target_real > 0
        if nonzero_mask.any():
            rel_err = (pred_real[nonzero_mask] - target_real[nonzero_mask]) / (target_real[nonzero_mask] + self.tau)
            L_relative = (rel_err ** 2).mean()
        else:
            L_relative = torch.zeros((), device=pred_log.device, dtype=pred_log.dtype)

        return self.lambda_log * L_log + self.lambda_relative * L_relative


# --loss CLI 인자로 loss를 선택하기 위한 레지스트리.
# 신규 loss 실험을 추가할 때는 클래스를 정의한 뒤 여기에 이름: 클래스 형태로만 등록하면 됨.
# 기존에 이미 코드/회의에서 다뤄진 loss(Pure MSE, MAE/L1, Huber, 기존/heavy-tail Weighted MSE,
# Huber+PINN, row/column total constraint)는 신규 후보로 다시 취급하지 않음 — weighted_mse만
# baseline으로 유지하고, 아래 5개(dual_scale/bin_balanced/cpc_hybrid/tweedie/tail_aware_relative)가
# 이번 실험의 신규 후보.
LOSS_REGISTRY = {
    'weighted_mse': WeightedMSELossWrapper,
    'dual_scale_mse': DualScaleMSELoss,
    'bin_balanced_mse': BinBalancedMSELoss,
    'cpc_hybrid': CPCHybridLoss,
    'tweedie_deviance': TweedieDevianceLoss,
    'tail_aware_relative': TailAwareRelativeLoss,
}