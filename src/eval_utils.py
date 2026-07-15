import contextlib
import os
import subprocess

import numpy as np
import pandas as pd
import torch


@contextlib.contextmanager
def eval_mode_for_validation(model):
    """
    검증/평가용 forward pass를 감싸는 컨텍스트 매니저. model.py는 전혀 건드리지 않고
    (모델 구조/가중치 불변), 실행 시점의 module 상태만 임시로 조정한다.

    - model.eval() 대신 model.train()을 유지: 기존 코드의 "PyTorch 우회용" 주석이 시사하는
      다른 PyTorch 이슈(커스텀 attention bias가 eval() fastpath에서 무시될 수 있는 문제로 추정)를
      피하기 위한 기존 관례를 그대로 유지함. 대신 nn.Dropout 서브모듈만 개별적으로 eval() 전환.
    - 추가로 nn.MultiheadAttention의 내부 dropout 확률을 검증 구간에서만 0으로 임시 설정:
      MPS 백엔드에서 no_grad() 추론 경로의 scaled_dot_product_attention이 dropout>0 조합을
      지원하지 않아 `NotImplementedError: scaled_dot_product_attention for MPS does not support
      dropout`이 발생함(CPU/CUDA에서는 발생하지 않는 MPS 전용 이슈로 확인됨,
      PYTORCH_ENABLE_MPS_FALLBACK=1로도 해결되지 않음 — 이 op 자체는 MPS에 구현되어 있고
      특정 인자 조합만 명시적으로 거부하는 것이라 fallback 대상이 아님).
      학습(model.train() 상태에서 실제 backward가 도는 구간)에는 영향 없음 — 이 컨텍스트를
      벗어나면 원래 dropout 확률로 정확히 복원됨.
    """
    dropout_modules = [m for m in model.modules() if isinstance(m, torch.nn.Dropout)]
    mha_modules = [m for m in model.modules() if isinstance(m, torch.nn.MultiheadAttention)]
    saved_mha_dropout = [m.dropout for m in mha_modules]

    model.train()
    for m in dropout_modules:
        m.eval()
    for m in mha_modules:
        m.dropout = 0.0
    try:
        yield
    finally:
        for m, d in zip(mha_modules, saved_mha_dropout):
            m.dropout = d


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


def get_git_commit_hash():
    try:
        repo_dir = os.path.dirname(os.path.abspath(__file__))
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=repo_dir, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return 'unknown'


def evaluate_breakdown(y_true_masked, y_pred_masked, O_idx_masked, D_idx_masked, run_meta, save_path,
                        diagnostic_full=None):
    """
    공식 비교 지표: 테스트 지역 관련(마스킹된) OD만을 대상으로 RMSE/MAE/CPC를 구간별로 계산해 CSV로 저장.

    y_true_masked, y_pred_masked, O_idx_masked, D_idx_masked: 1D real-scale numpy 배열.
        테스트 지역이 마스킹된 OD 항목만 대상으로 하며(공식 평가 범위), 서로 순서가 대응되어야 함.
    run_meta: dict, 결과 CSV에 함께 기록할 실험 메타데이터
        (pipeline, loss, seed, epochs, batch_size, alpha_info, git_commit, device 등)
    save_path: CSV 저장 경로
    diagnostic_full: 선택. (y_true_full, y_pred_full) 튜플 — (N,N) 전체 real-scale 배열.
        마스킹되지 않은 OD 정보까지 입력에 노출된 상태의 참고용 진단 지표이므로,
        공식 지표(evaluation_type='official_masked')와 섞이지 않도록
        evaluation_type='diagnostic_unmasked_input_exposed'인 별도 행(diagnostic_full_matrix)으로만 추가함.
    """
    y_true_masked = np.asarray(y_true_masked)
    y_pred_masked = np.asarray(y_pred_masked)
    O_idx_masked = np.asarray(O_idx_masked)
    D_idx_masked = np.asarray(D_idx_masked)
    is_same_dong = (O_idx_masked == D_idx_masked)

    # 주의: OD 값이 정수가 아닌 소수(예: 0.001)로도 존재해(실제 데이터의 상당 부분),
    # 구간 경계를 정수 기준(>=1, >=101 등)으로 잡으면 (0,1), (100,101), (999,1000) 사이의
    # 소수값이 어느 구간에도 속하지 못하고 누락된다. 0 / (0,100] / (100,1000) / [1000,∞)로
    # 겹침·빈틈 없는 완전 분할이 되도록 경계를 정의함.
    categories = {
        '전체 마스킹 평가 대상': np.ones_like(y_true_masked, dtype=bool),
        '실제값 0': y_true_masked == 0,
        '실제값 1~100': (y_true_masked > 0) & (y_true_masked <= 100),
        '실제값 101~999': (y_true_masked > 100) & (y_true_masked < 1000),
        '실제값 1000이상': y_true_masked >= 1000,
        '동일 행정동 내부 이동': is_same_dong,
        '서로 다른 행정동 간 이동': ~is_same_dong,
    }

    rows = []
    for name, sel in categories.items():
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
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
