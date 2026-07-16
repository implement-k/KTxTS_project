import contextlib
import os
import subprocess

import numpy as np
import pandas as pd
import torch


# 공식 평가 구간 이름 — 이 리스트가 유일한 정의처(single source of truth)이며,
# scripts/run_mae1_loss_search.py 등 다른 파일에서도 이 이름을 그대로 import해서 쓴다.
# 소수 OD 값이 존재하므로 "1~100"/"101~999"처럼 정수 구간 명칭을 쓰지 않는다.
OFFICIAL_CATEGORIES = [
    '전체 마스킹 평가 대상',
    '실제값 0',
    '실제값 0 초과 100 이하',
    '실제값 100 초과 1000 미만',
    '실제값 1000 이상',
    '동일 행정동 내부 이동(same-dong)',
    '서로 다른 행정동 간 이동(different-dong)',
]

ZERO_BIN_NAME = '실제값 0'
SMALL_OD_BIN_NAME = '실제값 0 초과 100 이하'
BIG_OD_BIN_NAME = '실제값 1000 이상'
OVERALL_NAME = '전체 마스킹 평가 대상'


@contextlib.contextmanager
def eval_mode_for_validation(model, device=None):
    """
    검증/평가용 forward pass를 감싸는 컨텍스트 매니저. model.py는 전혀 건드리지 않고
    (모델 구조/가중치 불변), 실행 시점의 module 상태만 임시로 조정한다.

    - model.eval() 대신 model.train()을 유지: 기존 코드의 "PyTorch 우회용" 주석이 시사하는
      다른 PyTorch 이슈(커스텀 attention bias가 eval() fastpath에서 무시될 수 있는 문제로 추정)를
      피하기 위한 기존 관례를 그대로 유지함(CUDA/CPU/MPS 공통). 대신 nn.Dropout 서브모듈만
      개별적으로 eval() 전환 — 이 부분은 historical baseline(CUDA)의 기존 동작과 동일하다.
    - nn.MultiheadAttention의 내부 dropout 확률을 0으로 임시 설정하는 것은 **device.type=='mps'
      일 때만** 적용한다: MPS 백엔드에서 no_grad() 추론 경로의 scaled_dot_product_attention이
      dropout>0 조합을 지원하지 않아 `NotImplementedError: scaled_dot_product_attention for MPS
      does not support dropout`이 발생하기 때문(CPU/CUDA에서는 발생하지 않는 MPS 전용 이슈로
      확인됨, PYTORCH_ENABLE_MPS_FALLBACK=1로도 해결 안 됨). 실제 실험은 Colab CUDA에서 수행하므로
      CUDA/CPU에서는 이 workaround가 전혀 적용되지 않아 historical baseline의 validation 동작이
      완전히 그대로 유지된다.
    - device를 안 넘기면(레거시 호출 호환) model의 파라미터 device로부터 자동 추론한다.
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device('cpu')

    dropout_modules = [m for m in model.modules() if isinstance(m, torch.nn.Dropout)]
    mha_modules = [m for m in model.modules() if isinstance(m, torch.nn.MultiheadAttention)]
    apply_mps_workaround = (device.type == 'mps')
    saved_mha_dropout = [m.dropout for m in mha_modules] if apply_mps_workaround else []

    model.train()
    for m in dropout_modules:
        m.eval()
    if apply_mps_workaround:
        for m in mha_modules:
            m.dropout = 0.0
    try:
        yield
    finally:
        if apply_mps_workaround:
            for m, d in zip(mha_modules, saved_mha_dropout):
                m.dropout = d


def check_validation_determinism(forward_fn, device, atol=1e-6):
    """
    동일 입력으로 검증용 forward pass를 두 번 실행해 값이 일치하는지 확인하는 작은 테스트.
    MPS/CUDA 각각에서 eval_mode_for_validation() 경로가 deterministic한지 확인하는 용도.

    forward_fn: 인자 없이 호출하면 forward 결과 텐서를 반환하는 콜러블
        (호출 측에서 model/입력 텐서를 이미 클로저로 캡처해서 넘겨줄 것).
    반환: dict(is_deterministic: bool, max_abs_diff: float, device: str)
    """
    with torch.no_grad():
        out1 = forward_fn()
        out2 = forward_fn()
    diff = (out1 - out2).abs()
    max_abs_diff = float(diff.max().item()) if diff.numel() else 0.0
    is_deterministic = max_abs_diff <= atol
    return {
        'is_deterministic': is_deterministic,
        'max_abs_diff': max_abs_diff,
        'device': str(device),
    }


def cpc_score(y_true, y_pred):
    numerator = 2 * np.sum(np.minimum(y_true, y_pred))
    denominator = np.sum(y_true) + np.sum(y_pred)
    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    cpc = float(cpc_score(y_true, y_pred))
    return {'rmse': rmse, 'mae': mae, 'cpc': cpc, 'n_samples': int(y_true.size)}


def compute_zero_bin_metrics(y_true_zero_true, y_pred_at_zero_true, threshold=10.0):
    """
    '실제값 0' 구간 전용 추가 지표. 이 구간에서는 CPC를 순위에 쓰지 않으므로(0/0 근처라 신호가
    약함) 대신 아래를 기록한다.
    - rmse, mae: 일반 지표(참고용)
    - mean_pred: 평균 예측량(0이어야 이상적)
    - false_positive_rate: 실제 0인데 예측이 명확히 양수(>0.5)로 나온 비율
    - exceed_threshold_rate: 예측값이 threshold(기본 10)를 넘은 비율(더 심각한 오탐 비율)
    """
    y_pred_at_zero_true = np.asarray(y_pred_at_zero_true)
    n = y_pred_at_zero_true.size
    if n == 0:
        return {'rmse': float('nan'), 'mae': float('nan'), 'mean_pred': float('nan'),
                'false_positive_rate': float('nan'), 'exceed_threshold_rate': float('nan'),
                'threshold': threshold, 'n_samples': 0}
    y_true_zero = np.zeros_like(y_pred_at_zero_true)
    rmse = float(np.sqrt(np.mean((y_true_zero - y_pred_at_zero_true) ** 2)))
    mae = float(np.mean(np.abs(y_pred_at_zero_true)))
    mean_pred = float(y_pred_at_zero_true.mean())
    false_positive_rate = float((y_pred_at_zero_true > 0.5).mean())
    exceed_threshold_rate = float((y_pred_at_zero_true > threshold).mean())
    return {
        'rmse': rmse, 'mae': mae, 'mean_pred': mean_pred,
        'false_positive_rate': false_positive_rate, 'exceed_threshold_rate': exceed_threshold_rate,
        'threshold': threshold, 'n_samples': int(n),
    }


def get_git_commit_hash():
    try:
        repo_dir = os.path.dirname(os.path.abspath(__file__))
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=repo_dir, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return 'unknown'


def evaluate_breakdown(y_true_masked, y_pred_masked, O_idx_masked, D_idx_masked, run_meta, save_path,
                        diagnostic_full=None, zero_bin_threshold=10.0):
    """
    공식 비교 지표: strict/legacy protocol에서 정의된 validation(또는 최종 test) 마스킹 대상만을
    기준으로 RMSE/MAE/CPC를 구간별로 계산해 CSV로 저장. run_meta에 'protocol' 키를 반드시
    포함해야 legacy/strict 결과가 CSV에서 섞이지 않는다(호출 측 책임).

    y_true_masked, y_pred_masked, O_idx_masked, D_idx_masked: 1D real-scale numpy 배열.
        평가 대상 마스킹된 OD 항목만 대상으로 하며, 서로 순서가 대응되어야 함.
    run_meta: dict, 결과 CSV에 함께 기록할 실험 메타데이터
        (protocol, pipeline, loss, seed, epochs, batch_size, git_commit, device 등)
    save_path: CSV 저장 경로
    diagnostic_full: 선택. (y_true_full, y_pred_full) 튜플 — (N,N) 전체 real-scale 배열.
        마스킹되지 않은 OD 정보까지 입력에 노출된 상태의 참고용 진단 지표이므로,
        공식 지표(evaluation_type='official_masked')와 섞이지 않도록
        evaluation_type='diagnostic_unmasked_input_exposed'인 별도 행(diagnostic_full_matrix)으로만 추가함.
    zero_bin_threshold: '실제값 0' 구간의 exceed_threshold_rate 계산에 쓰는 임계값(기본 10).
    """
    y_true_masked = np.asarray(y_true_masked)
    y_pred_masked = np.asarray(y_pred_masked)
    O_idx_masked = np.asarray(O_idx_masked)
    D_idx_masked = np.asarray(D_idx_masked)
    is_same_dong = (O_idx_masked == D_idx_masked)

    # 주의: OD 값이 정수가 아닌 소수(예: 0.001)로도 존재해(실제 데이터의 상당 부분),
    # 구간 경계를 정수 기준(>=1, >=101 등)으로 잡으면 (0,1), (100,101), (999,1000) 사이의
    # 소수값이 어느 구간에도 속하지 못하고 누락된다. 0 / (0,100] / (100,1000) / [1000,∞)로
    # 겹침·빈틈 없는 완전 분할이 되도록 경계를 정의함. 표시 이름도 "1~100" 같은 정수 구간
    # 명칭이 아니라 실제 부등식 그대로 표기한다.
    categories = {
        OVERALL_NAME: np.ones_like(y_true_masked, dtype=bool),
        ZERO_BIN_NAME: y_true_masked == 0,
        SMALL_OD_BIN_NAME: (y_true_masked > 0) & (y_true_masked <= 100),
        '실제값 100 초과 1000 미만': (y_true_masked > 100) & (y_true_masked < 1000),
        BIG_OD_BIN_NAME: y_true_masked >= 1000,
        '동일 행정동 내부 이동(same-dong)': is_same_dong,
        '서로 다른 행정동 간 이동(different-dong)': ~is_same_dong,
    }
    assert list(categories.keys()) == OFFICIAL_CATEGORIES, "categories와 OFFICIAL_CATEGORIES 정의가 어긋남"

    rows = []
    for name, sel in categories.items():
        if name == ZERO_BIN_NAME:
            # 실제값 0 구간: CPC는 순위에 쓰지 않으므로 계산하지 않고, 대신 0-bin 전용 지표를 기록.
            zb = compute_zero_bin_metrics(y_true_masked[sel], y_pred_masked[sel], threshold=zero_bin_threshold)
            rows.append({'category': name, 'evaluation_type': 'official_masked',
                         'rmse': zb['rmse'], 'mae': zb['mae'], 'cpc': float('nan'),
                         'n_samples': zb['n_samples'], 'mean_pred': zb['mean_pred'],
                         'false_positive_rate': zb['false_positive_rate'],
                         'exceed_threshold_rate': zb['exceed_threshold_rate'],
                         'zero_bin_threshold': zb['threshold'],
                         **run_meta})
            continue
        if sel.sum() == 0:
            metrics = {'rmse': float('nan'), 'mae': float('nan'), 'cpc': float('nan'), 'n_samples': 0}
        else:
            metrics = compute_metrics(y_true_masked[sel], y_pred_masked[sel])
        rows.append({'category': name, 'evaluation_type': 'official_masked', **metrics, **run_meta})

    if diagnostic_full is not None:
        y_true_full, y_pred_full = diagnostic_full
        diag_metrics = compute_metrics(
            np.asarray(y_true_full).flatten(), np.asarray(y_pred_full).flatten()
        )
        rows.append({
            'category': 'diagnostic_full_matrix',
            'evaluation_type': 'diagnostic_unmasked_input_exposed',
            **diag_metrics,
            **run_meta,
        })

    df = pd.DataFrame(rows)
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"Evaluation breakdown saved to {save_path}")
    return df


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
