import torch
import torch.nn.functional as F
import torch.nn as nn


class HuberLossWrapper(nn.Module):
    """
    Pure Huber Loss. 이미 과거에 시도되어 실패로 분류된 loss(1차 loss search 후보 아님).
    참고용으로만 남겨둠 — LOSS_REGISTRY/실행 후보 목록에는 포함하지 않는다.
    """
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta

    def forward(self, pred_log, target_log, mask):
        loss = F.huber_loss(pred_log[mask], target_log[mask], delta=self.delta, reduction='mean')
        return loss


class DynamicWeightedMSELoss(nn.Module):
    """
    Historical MAE1 standalone baseline (git a17b361, 2026-07-10 15:12 KST, author implement-k,
    "[수정] loss함수 큰 숫자 범위에서도 잘 맞추도록 수정" 커밋에서 복원).

    weight = 1 + alpha * expm1(target_log)      (real-scale target 기준 가중치)
    weight = weight / (weight.mean() + eps)      (배치 내 평균 1로 자기정규화)
    loss   = mean((pred_log - target_log)^2 * weight)

    alpha는 학습 루프에서 epoch마다 외부에서 갱신하는 것을 전제로 한다(원본 코드와 동일한 방식):
        progress = epoch / max(1, total_epochs - 1)
        current_alpha = max(1.0, 10.0 * (1.0 - progress))   # 10.0 -> 1.0으로 감소
        criterion.alpha = current_alpha  # 매 epoch 대입

    주의(복원 신뢰도): 이 코드는 git에 남아있는 가장 가까운 형태를 그대로 복원한 것이며,
    노션에 기록된 CPC 0.5434 / RMSE 924.49가 정확히 "이 커밋의 코드"로 나온 결과라는 보장은
    없다(그 시점 코드와 실제 실행 로그가 git에 함께 남아있지 않음). learning_rate에 대해서도
    노션 기록은 1e-3이라 하나, 이 커밋 시점의 실제 코드는 AdamW(lr=1e-4)를 사용하고 있어
    두 기록이 서로 다르다 — 자세한 내용은 문서화된 보고 참고.
    """
    def __init__(self, alpha=10.0, eps=1e-8):
        super().__init__()
        self.alpha = alpha
        self.eps = eps

    def forward(self, pred_log, target_log, mask=None):
        if mask is not None:
            pred_log = pred_log[mask]
            target_log = target_log[mask]

        real_target = torch.expm1(target_log)
        weight = 1.0 + self.alpha * real_target
        weight = weight / (weight.mean() + self.eps)
        loss = ((pred_log - target_log) ** 2) * weight
        return loss.mean()


class WeightedMSELossWrapper(nn.Module):
    """
    alpha 고정 버전. **historical baseline이 아니다** — 위 DynamicWeightedMSELoss가 historical
    baseline이며, 이 클래스는 참고용 후보(weighted_mse_fixed)로만 등록한다.
    가중치를 log1p-scale target 기준으로 직접 계산하며 배치 자기정규화가 없다(단순한 형태).
    """
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
    log1p 공간 MSE(L_log) + real-scale MSE(L_real, global_scale^2로 정규화)를 결합.

    L_log  = MSE(pred_log, target_log)
    L_real = mean( ((pred_real - target_real) / global_scale) ** 2 )   [= MSE / global_scale^2]
    L      = lambda_log * L_log + (1 - lambda_log) * L_real

    - pred_real = expm1(clamp(pred_log, max=pred_log_clip))  — **하한 clamp를 두지 않는다.**
      expm1은 정의역 전체에서 매끈하고 하한이 자연스럽게 -1로 bounded되며, gradient(=exp(pred_log))가
      pred_log<0 구간에서도 항상 0보다 크므로 학습 신호가 끊기지 않는다. 이전 구현의
      `clamp(min=0)`은 pred_log<0일 때 gradient를 완전히 죽이는 문제가 있어 제거했다.
      상한만 pred_log_clip으로 clamp해 overflow(inf)를 방지한다.
    - global_scale: target_real의 robust한 크기(RMS) 기준. 생성자에 넘기면(target_distribution
      분석의 전역 통계) 그 값을 고정으로 쓰고, 없으면 배치별 detached RMS로 폴백한다(gradient 차단).
    - forward 호출 후 self.last_log_term / last_real_term / last_total 에 각 항의 크기(detached
      스칼라)가 기록되므로, smoke test에서 이를 읽어 로그로 남길 수 있다.
    """
    def __init__(self, lambda_log=0.5, global_scale=None, eps=1e-2, min_scale=1.0, pred_log_clip=20.0):
        super().__init__()
        assert 0.0 <= lambda_log <= 1.0, "lambda_log는 [0,1] 범위여야 함"
        self.lambda_log = lambda_log
        self.eps = eps
        self.min_scale = min_scale
        self.pred_log_clip = pred_log_clip
        gs = float(global_scale) if global_scale is not None else -1.0
        self.register_buffer('global_scale', torch.tensor(gs))
        self.last_log_term = None
        self.last_real_term = None
        self.last_total = None

    def forward(self, pred_log, target_log, mask=None):
        if mask is not None:
            pred_log = pred_log[mask]
            target_log = target_log[mask]

        L_log = F.mse_loss(pred_log, target_log, reduction='mean')

        pred_log_clamped = torch.clamp(pred_log, max=self.pred_log_clip)  # 상한만 clamp(overflow 방지)
        pred_real = torch.expm1(pred_log_clamped)  # 하한 clamp 없음 -> gradient 항상 유지
        target_real = torch.expm1(target_log)

        if self.global_scale.item() > 0:
            scale = self.global_scale
        else:
            scale = torch.sqrt(torch.mean(target_real.detach() ** 2) + self.eps)
        scale = torch.clamp(scale, min=self.min_scale).detach()

        rel_err = (pred_real - target_real) / scale
        L_real = (rel_err ** 2).mean()  # MSE(pred_real, target_real) / scale^2 와 동일

        total = self.lambda_log * L_log + (1.0 - self.lambda_log) * L_real

        self.last_log_term = L_log.detach()
        self.last_real_term = L_real.detach()
        self.last_total = total.detach()
        return total


class BinBalancedMSELoss(nn.Module):
    """
    통행량 크기 구간(0 / 0<x<=100 / 100<x<1000 / x>=1000)별 inverse-sqrt-frequency 가중치를
    적용한 log1p 공간 MSE. bin_freq는 "실제 masking sampler(커리큘럼)가 선택하는 loss 대상"의
    분포를 고정 seed로 충분히 샘플링해 추정한 값을 주입받는 것을 전제로 한다(전체 train x train
    단순 flatten 분포가 아님 — scripts/analyze_target_distribution.py --sampled 참고).
    strict protocol에서는 train-only(strict) loss mask 기준으로 다시 산출한 bin_freq를 쓴다.

    정규화(기대값 1이 되도록):
        w_b = 1 / sqrt(f_b)
        w_b = w_b / sum_b(f_b * w_b)     =>  E_f[w] = 1
    배치마다 다시 평균 1로 재정규화하지 않는다(배치 구성에 따라 가중치가 흔들리지 않도록).
    weight 상한(max_weight)은 명시적으로 설정하며, 실행 metadata(run_meta/experiment_config)에
    함께 기록해야 한다.
    """
    def __init__(self, bin_freq, max_weight=10.0, eps=1e-6):
        super().__init__()
        bin_freq_t = torch.clamp(torch.tensor(bin_freq, dtype=torch.float32), min=eps)
        assert bin_freq_t.numel() == 4, "bin_freq는 [0, 0<x<=100, 100<x<1000, x>=1000] 4개 구간이어야 함"

        raw_w = 1.0 / torch.sqrt(bin_freq_t)
        expectation = (bin_freq_t * raw_w).sum()
        w = raw_w / expectation
        w = torch.clamp(w, max=max_weight)

        self.register_buffer('bin_weights', w)  # (4,)
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

        se = (pred_log - target_log) ** 2
        return (se * weight).mean()


class CPCHybridLoss(nn.Module):
    """
    log1p 공간 MSE + (1 - ExactCPC)를 결합. 최종 공식 평가 지표인 CPC를 직접 겨냥.

    용어 정정: 아래 min 연산은 온도(temperature) 파라미터를 쓰는 "smooth approximation"이
    아니라, min(a,b) = (a+b-|a-b|)/2 라는 **정확한(exact) 항등식**을 그대로 계산한 것이다.
    이 식은 a=b인 지점(|.|의 미분 불연속점)에서만 subgradient가 되고 그 외에는 exact gradient이며,
    온도 기반 근사보다 수치적으로 더 안정적이라 이 방식을 택했다. 따라서 "ExactCPC /
    abs 기반 exact-min-subgradient"로 부르는 것이 정확하다.

    ExactCPC = 2 * sum(exact_min(pred_real, target_real)) / (sum(pred_real + target_real) + eps)
    exact_min(a, b) = (a + b - |a - b|) / 2
    L = MSE(pred_log, target_log) + lambda_cpc * (1 - ExactCPC)

    - pred_real = expm1(clamp(pred_log, max=pred_log_clip)) — 하한 clamp 없음(DualScaleMSELoss와
      동일한 이유로 pred_log<0 구간 gradient를 죽이지 않기 위함).
    - scale 특성: ExactCPC는 분자/분모 모두 pred_real+target_real의 합에 비례해 커지는 비율량이라
      원리적으로 배치 크기·target 총량에 대해 scale-invariant하다. 다만 한 배치의 마스킹 크기가
      매우 작으면(예: k=1) 표본이 적어 CPC 추정 자체의 분산이 커질 수 있음 — 이는 지표의 성질이지
      코드 버그는 아니며, smoke test 로그에 마스크 크기와 함께 기록해 확인한다.
    - 퇴행적 해 확인: pred_real을 전체적으로 키우면(>= target_real인 영역) exact_min이
      target_real로 고정되는 반면 분모(pred+target 합)는 커지므로 ExactCPC는 오히려 감소한다.
      즉 "예측 총량을 무조건 키워 CPC를 속이는" 해는 이 식에서 이득이 되지 않으며,
      MSE 항이 추가로 이를 억제한다.
    - 1차 loss search에서는 lambda_cpc=0.1 하나만 사용한다.
    """
    def __init__(self, lambda_cpc=0.1, eps=1e-6, pred_log_clip=20.0):
        super().__init__()
        assert lambda_cpc >= 0.0, "lambda_cpc는 0 이상이어야 함"
        self.lambda_cpc = lambda_cpc
        self.eps = eps
        self.pred_log_clip = pred_log_clip
        self.last_mse_term = None
        self.last_cpc_term = None

    def forward(self, pred_log, target_log, mask=None):
        if mask is not None:
            pred_log = pred_log[mask]
            target_log = target_log[mask]

        L_mse = F.mse_loss(pred_log, target_log, reduction='mean')

        pred_log_clamped = torch.clamp(pred_log, max=self.pred_log_clip)
        pred_real = torch.expm1(pred_log_clamped)  # 하한 clamp 없음 -> gradient 유지
        target_real = torch.expm1(target_log)

        exact_min = (pred_real + target_real - torch.abs(pred_real - target_real)) / 2.0
        numerator = 2.0 * exact_min.sum()
        denominator = (pred_real + target_real).sum() + self.eps
        exact_cpc = numerator / denominator

        cpc_term = self.lambda_cpc * (1.0 - exact_cpc)
        self.last_mse_term = L_mse.detach()
        self.last_cpc_term = cpc_term.detach()
        return L_mse + cpc_term


class TweedieDevianceLoss(nn.Module):
    """
    [EXPERIMENTAL — 1차 loss search 실행 후보에서 제외]

    Tweedie deviance loss (1 < p < 2). 클래스는 참고/후속 실험용으로 남겨두되, 현재 MAE1 출력
    구조(log1p target을 직접 예측, positive-mean parameterization을 위한 명시적 output link 없음)와
    맞지 않아 이번 1차 실행 후보에는 포함하지 않는다.

    제외 이유: mu(=예측 평균, 반드시 양수)를 얻으려면 pred_log를 expm1한 뒤 양수로 만들어야 하는데,
    clamp(expm1(pred_log), min=eps)는 pred_log가 음수(=예측이 0 미만)인 구간에서 gradient를
    완전히 죽인다. 이를 근본적으로 고치려면 모델 출력층에 Softplus 등 양수 보장 link를 추가하고
    평가 시 그에 맞는 역변환도 함께 재설계해야 하는데, 이는 모델 구조 변경 없이 loss만 바꾸는
    이번 1차 실험의 범위를 벗어난다. Softplus 하나만 임시로 끼워 넣어 억지로 돌리지 않기로 함.

    d(y, mu) = 2 * ( y^(2-p)/((1-p)(2-p)) - y*mu^(1-p)/(1-p) + mu^(2-p)/(2-p) )
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
        mu = torch.clamp(torch.expm1(pred_log_clamped), min=self.eps)  # 여전히 gradient 문제 있음 - 실험 제외 사유 참고

        term1 = torch.pow(y, 2 - p) / ((1 - p) * (2 - p))
        term2 = y * torch.pow(mu, 1 - p) / (1 - p)
        term3 = torch.pow(mu, 2 - p) / (2 - p)

        deviance = 2.0 * (term1 - term2 + term3)
        return deviance.mean()


class PositiveRelativeHybridLoss(nn.Module):
    """
    (구 TailAwareRelativeLoss 개명 — "tail"이 아니라 target_real > 0인 **모든 양수값**에
    상대오차를 적용하므로 이름을 실제 동작에 맞게 수정함)

    작은 값에는 log1p-MSE(사실상 절대오차 성격)를, 모든 양수 값에는 상대오차를 추가로 반영.

    relative_error = (pred_real - target_real) / (target_real + tau)   [target_real > 0 인 항목만]
    L = lambda_log * MSE(pred_log, target_log) + lambda_relative * mean(relative_error^2)

    - pred_real = expm1(clamp(pred_log, max=pred_log_clip)) — 하한 clamp 없음(다른 신규 loss와
      동일하게 pred_log<0 구간 gradient를 죽이지 않기 위함).
    - target_real == 0인 항목은 상대오차 항 계산에서 제외한다(0 근처 폭발 방지, 대신 L_log가
      이 항목들의 학습을 담당).
    - MAPE(평균절대백분율오차)를 그대로 쓰지 않고, tau로 분모를 이동시킨 완화된 상대오차의
      "제곱평균"을 사용한다.
    - tau=50: target_distribution 분석 결과 양수 OD median이 4.335로 매우 작아, tau를 median에
      가깝게 두면(예: 4.335) median 근처 값들에서도 상대오차가 과도하게 커진다
      (분모가 target_real과 비슷한 크기라 상대오차의 민감도가 target_real≈tau 부근에서 급격해짐).
      tau=50은 median(4.34)~99%ile(727) 사이에서 작은 값 쪽의 민감도를 눌러주면서도, target_real이
      tau보다 충분히 커지는 큰 OD 구간에서는 여전히 상대오차 성격을 유지한다.
      (tau=25는 2차 후보로 남겨두되 1차 실행에는 포함하지 않음.)
    """
    def __init__(self, lambda_log=0.7, lambda_relative=0.3, tau=50.0, pred_log_clip=20.0):
        super().__init__()
        assert 0.0 <= lambda_log, "lambda_log는 0 이상이어야 함"
        assert lambda_relative >= 0.0, "lambda_relative는 0 이상이어야 함"
        assert tau > 0.0, "tau는 0보다 커야 함"
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
        pred_real = torch.expm1(pred_log_clamped)  # 하한 clamp 없음
        target_real = torch.expm1(target_log)

        nonzero_mask = target_real > 0
        if nonzero_mask.any():
            rel_err = (pred_real[nonzero_mask] - target_real[nonzero_mask]) / (target_real[nonzero_mask] + self.tau)
            L_relative = (rel_err ** 2).mean()
        else:
            L_relative = torch.zeros((), device=pred_log.device, dtype=pred_log.dtype)

        return self.lambda_log * L_log + self.lambda_relative * L_relative


# 하위 호환 별칭(과거 이름을 참조하는 코드가 있을 경우 대비) — 신규 코드에서는 PositiveRelativeHybridLoss를 사용할 것.
TailAwareRelativeLoss = PositiveRelativeHybridLoss


# --loss CLI 인자로 loss를 선택하기 위한 레지스트리.
#
# baseline:
#   dynamic_weighted_mse : historical MAE1 standalone baseline (git a17b361 복원, alpha 10->1 schedule)
#   weighted_mse_fixed   : 참고 후보. historical baseline이 아님(alpha=1.5 고정)
#
# 신규 후보(1차 loss search 대상):
#   dual_scale_mse, bin_balanced_mse, cpc_hybrid, positive_relative_hybrid
#
# 제외:
#   huber                : 이미 과거에 실패로 분류됨(재실행 대상 아님, 참고용으로만 등록)
#   tweedie_deviance      : 1차 실행 후보 제외(TweedieDevianceLoss 클래스 docstring 참고), 실험용으로만 등록
LOSS_REGISTRY = {
    'dynamic_weighted_mse': DynamicWeightedMSELoss,
    'weighted_mse_fixed': WeightedMSELossWrapper,
    'dual_scale_mse': DualScaleMSELoss,
    'bin_balanced_mse': BinBalancedMSELoss,
    'cpc_hybrid': CPCHybridLoss,
    'positive_relative_hybrid': PositiveRelativeHybridLoss,
    # 아래 둘은 등록만 되어 있고 오케스트레이터의 1차 실행 후보 목록에는 포함하지 않는다.
    'huber': HuberLossWrapper,
    'tweedie_deviance': TweedieDevianceLoss,
}
