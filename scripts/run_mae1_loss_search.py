#!/usr/bin/env python3
"""
SpatialODMAE1(mae1) loss 탐색 자동 실험 오케스트레이터 (v2).

Stage 0: Historical baseline reproduction (legacy protocol, dynamic_weighted_mse, 50 epoch)
Stage 1: Runtime smoke (strict protocol, 각 후보 20~50 step, 순위에 미사용)
Stage 2: Screening (strict protocol, baseline+smoke 통과 후보, planned 50 epoch 중 stop_epoch=15)
Stage 3: Final (strict protocol, baseline + screening 상위 1~2개, 50 epoch, 최종 test는 여기서만)

이 스크립트가 직접 건드리는 것은 "어떤 loss/하이퍼파라미터/protocol로 몇 epoch 학습할지"뿐이며,
실제 학습/평가 로직(src/model/train.py, model_test.py, models.py, dataset.py, eval_utils.py)은
subprocess로 그대로 호출만 한다.

resume: 완료 판단은 checkpoint/CSV 존재만으로 하지 않는다.
- smoke: run_dir/smoke_diagnostics.json 존재 여부
- 조기종료(screening) 실행: run_dir/training_state.pt의 current_epoch/identity로 판단
- 전체 완료(Stage0/Stage3): run_dir/completed.json의 identity가 현재 실행 조건과 정확히 일치할 때만
재시도 로그는 attempt_01/train.log, attempt_02/train.log로 분리해, 이전 실패 로그의
Traceback/NaN이 다음 시도의 성공 판정을 오염시키지 않게 한다.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from eval_utils import OFFICIAL_CATEGORIES, OVERALL_NAME, SMALL_OD_BIN_NAME, BIG_OD_BIN_NAME, ZERO_BIN_NAME  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PY = os.path.join(REPO_ROOT, 'src', 'model', 'train.py')

HISTORICAL_CPC = 0.5434
HISTORICAL_RMSE = 924.49

CANDIDATE_DEFS = [
    # baseline: historical MAE1 standalone (git a17b361 복원, alpha 10->1 schedule)
    {'label': 'dynamic_weighted_mse_baseline', 'loss': 'dynamic_weighted_mse', 'params': {}, 'is_baseline': True},
    # 신규 후보(1차 loss search 대상)
    {'label': 'dual_scale_mse', 'loss': 'dual_scale_mse', 'params': {'lambda_log': 0.5}, 'is_baseline': False},
    {'label': 'bin_balanced_mse_inv_sqrt', 'loss': 'bin_balanced_mse', 'params': {}, 'is_baseline': False},
    {'label': 'cpc_hybrid', 'loss': 'cpc_hybrid', 'params': {'lambda_cpc': 0.1}, 'is_baseline': False},
    {'label': 'positive_relative_hybrid', 'loss': 'positive_relative_hybrid',
     'params': {'lambda_log': 0.7, 'lambda_relative': 0.3, 'tau': 50.0}, 'is_baseline': False},
    # 참고 후보(historical baseline 아님, 순위 안전조건에는 포함하되 "추천"으로는 선택 안 함 별도 표기)
    {'label': 'weighted_mse_fixed_reference', 'loss': 'weighted_mse_fixed', 'params': {'alpha': 1.5}, 'is_baseline': False, 'is_reference': True},
]

PLANNED_EPOCHS = 50
SCREENING_STOP_EPOCH = 15
MAIN_SEED = 42
SMOKE_SEED = 999
SMOKE_STEPS = 30


def inject_derived_params(candidate, derived):
    params = dict(candidate['params'])
    if candidate['loss'] == 'dual_scale_mse':
        params['global_scale'] = derived['dual_scale_mse']['global_scale']
    elif candidate['loss'] == 'bin_balanced_mse':
        params['bin_freq'] = derived['bin_balanced_mse']['bin_freq']
    return params


def log(msg):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    return line


def append_run_status(run_status_path, lines):
    with open(run_status_path, 'a', encoding='utf-8') as f:
        for line in lines:
            f.write(line + '\n')


def get_git_commit_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT).decode().strip()
    except Exception:
        return 'unknown'


def canonical_json(params):
    return json.dumps(params, sort_keys=True, ensure_ascii=False)


def capture_env_info(python_exe):
    info = {'python_exe': python_exe}
    try:
        out = subprocess.check_output([
            python_exe, '-c',
            "import torch,platform,json;"
            "print(json.dumps({'torch_version': torch.__version__,"
            "'mps_built': torch.backends.mps.is_built(),"
            "'mps_available': torch.backends.mps.is_available(),"
            "'cuda_available': torch.cuda.is_available(),"
            "'cuda_device_name': (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),"
            "'platform': platform.platform(),"
            "'machine': platform.machine()}))"
        ]).decode().strip()
        info.update(json.loads(out))
    except Exception as e:
        info['error'] = str(e)
    return info


# ---------------------------------------------------------------------------
# 완료 판단(resume 핵심): checkpoint/CSV 존재만으로 판단하지 않는다.
# ---------------------------------------------------------------------------
def build_identity(model, loss, canonical_params_json, seed, epochs, batch_size, protocol, git_commit):
    return {
        'protocol': protocol, 'loss': loss, 'canonical_loss_params': canonical_params_json,
        'seed': seed, 'planned_epochs': epochs, 'batch_size': batch_size, 'model': model,
        'git_commit': git_commit,
    }


def identity_matches(record, identity):
    for key, expected in identity.items():
        actual = record.get(key)

        # 기존 completed.json은 planned_epochs 대신 epochs로 저장되어 있다.
        if key == 'planned_epochs' and actual is None:
            actual = record.get('epochs')

        if actual != expected:
            return False

    return True


def check_smoke_complete(run_dir):
    path = os.path.join(run_dir, 'smoke_diagnostics.json')
    if not os.path.exists(path):
        return False, None
    with open(path, 'r', encoding='utf-8') as f:
        diag = json.load(f)
    ok = (diag.get('total_steps_done', 0) >= diag.get('requested_smoke_steps', 1)) and not diag.get('nan_or_inf_grad_norm', True)
    return ok, diag


def check_full_complete(run_dir, identity):
    path = os.path.join(run_dir, 'completed.json')
    if not os.path.exists(path):
        return False, None
    with open(path, 'r', encoding='utf-8') as f:
        completed = json.load(f)
    ok = identity_matches(completed, identity) and completed.get('returncode') == 0
    return ok, completed


def check_screening_progress(run_dir, identity, target_epoch):
    """training_state.pt의 current_epoch/identity로 조기종료(screening) 목표에 도달했는지 확인."""
    path = os.path.join(run_dir, 'training_state.pt')
    if not os.path.exists(path):
        return False, None
    try:
        import torch
        state = torch.load(path, map_location='cpu', weights_only=False)
    except Exception:
        return False, None
    ok = identity_matches(state, identity) and state.get('current_epoch', -1) >= (target_epoch - 1)
    return ok, {'current_epoch': state.get('current_epoch')}


def scan_log_for_instability(log_path):
    if not os.path.exists(log_path):
        return True, 'log 파일이 생성되지 않음'
    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()
    if 'Traceback (most recent call last)' in text:
        tail = text.strip().splitlines()[-15:]
        return True, 'Python traceback: ' + ' | '.join(tail)
    import re
    if re.search(r"\bnan\b", text, re.IGNORECASE) or re.search(r"[^a-zA-Z]inf[^a-zA-Z]", text):
        return True, 'nan/inf 문자열이 로그에서 발견됨'
    return False, None


def run_train(python_exe, run_dir, model, loss, params, seed, protocol, epochs,
              stop_epoch=None, smoke_steps=None, run_final_test=False, batch_size=32,
              max_retries=2, timing_csv=None, failures_csv=None):
    """
    run_dir에서 train.py를 실행한다(resume-aware). 각 attempt는 run_dir/attempt_NN/train.log에
    기록되어 이전 실패 로그가 다음 시도의 실패 판정을 오염시키지 않는다.
    checkpoint/training_state.pt/completed.json 등은 run_dir 루트에 남아 attempt 간 공유된다
    (즉 attempt 2는 attempt 1이 남긴 training_state.pt를 이어받아 resume한다).
    """
    os.makedirs(run_dir, exist_ok=True)
    canon = canonical_json(params)
    git_commit = get_git_commit_hash()
    identity = build_identity(model, loss, canon, seed, epochs, batch_size, protocol, git_commit)

    # --- 이미 완료됐는지 확인 ---
    if smoke_steps is not None:
        ok, info = check_smoke_complete(run_dir)
        if ok:
            log(f"[skip] {run_dir} 이미 smoke 완료됨")
            return True, info
    elif stop_epoch is not None and stop_epoch < epochs:
        ok, info = check_screening_progress(run_dir, identity, stop_epoch)
        if ok:
            log(f"[skip] {run_dir} 이미 stop_epoch={stop_epoch}까지 도달함")
            return True, info
    else:
        ok, info = check_full_complete(run_dir, identity)
        if ok:
            log(f"[skip] {run_dir} 이미 완전히 완료됨(completed.json 일치)")
            return True, info

    cmd = [
        python_exe, TRAIN_PY,
        '--model', model, '--epochs', str(epochs), '--batch_size', str(batch_size),
        '--loss', loss, '--loss-params', json.dumps(params), '--seed', str(seed),
        '--protocol', protocol,
    ]
    if stop_epoch is not None:
        cmd += ['--stop-epoch', str(stop_epoch)]
    if smoke_steps is not None:
        cmd += ['--smoke-steps', str(smoke_steps)]
    if run_final_test:
        cmd += ['--run-final-test']

    attempt = 0
    while True:
        attempt += 1
        attempt_dir = os.path.join(run_dir, f'attempt_{attempt:02d}')
        os.makedirs(attempt_dir, exist_ok=True)
        log_path = os.path.join(attempt_dir, 'train.log')

        log(f"[{os.path.basename(run_dir)}] attempt {attempt}/{max_retries + 1} 시작: {' '.join(cmd)}")
        start = time.time()
        try:
            with open(log_path, 'w', encoding='utf-8') as logf:  # 새 attempt는 항상 새 로그 파일(이전 실패 오염 방지)
                proc = subprocess.run(cmd, cwd=run_dir, stdout=logf, stderr=subprocess.STDOUT)
            returncode = proc.returncode
        except Exception as e:
            returncode = -1
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(f"\n=== 오케스트레이터 예외: {e}\n{traceback.format_exc()}\n")
        elapsed = time.time() - start

        unstable, reason = scan_log_for_instability(log_path)

        if smoke_steps is not None:
            ok, info = check_smoke_complete(run_dir)
        elif stop_epoch is not None and stop_epoch < epochs:
            ok, info = check_screening_progress(run_dir, identity, stop_epoch)
        else:
            ok, info = check_full_complete(run_dir, identity)

        success = (returncode == 0) and (not unstable) and ok

        if timing_csv:
            _append_csv_row(timing_csv, ['run_dir', 'attempt', 'wall_time_sec', 'returncode', 'success'],
                             {'run_dir': run_dir, 'attempt': attempt, 'wall_time_sec': round(elapsed, 2),
                              'returncode': returncode, 'success': success})

        if success:
            epoch_time = elapsed / max(1, (stop_epoch if (stop_epoch is not None and stop_epoch < epochs) else epochs))
            log(f"[{os.path.basename(run_dir)}] 성공 ({elapsed:.1f}s, epoch당 약 {epoch_time:.1f}s)")
            return True, {'info': info, 'elapsed': elapsed, 'epoch_time': epoch_time}
        else:
            error_summary = reason or f'returncode={returncode}, check_ok={ok}'
            if failures_csv:
                _append_csv_row(failures_csv, ['run_dir', 'attempt', 'error_summary', 'log_path'],
                                 {'run_dir': run_dir, 'attempt': attempt, 'error_summary': error_summary,
                                  'log_path': log_path})
            log(f"[{os.path.basename(run_dir)}] 실패 (attempt {attempt}): {error_summary}")
            if attempt > max_retries:
                log(f"[{os.path.basename(run_dir)}] 최종 실패 — 이후 stage에서 제외")
                return False, {'error': error_summary}


def _append_csv_row(path, fields, row):
    import csv
    is_new = not os.path.exists(path)
    with open(path, 'a', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def load_official_metrics(csv_path):
    if not os.path.exists(csv_path):
        return None
    import csv
    metrics = {}
    with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('evaluation_type') != 'official_masked':
                continue

            def _f(v):
                try:
                    return float(v) if v not in ('', 'nan', None) else float('nan')
                except ValueError:
                    return float('nan')
            metrics[row.get('category')] = {
                'rmse': _f(row.get('rmse')), 'mae': _f(row.get('mae')), 'cpc': _f(row.get('cpc')),
                'n_samples': int(float(row['n_samples'])) if row.get('n_samples') not in ('', None) else 0,
                'mean_pred': _f(row.get('mean_pred')),
                'false_positive_rate': _f(row.get('false_positive_rate')),
                'exceed_threshold_rate': _f(row.get('exceed_threshold_rate')),
            }
    return metrics if metrics else None


# ---------------------------------------------------------------------------
# 다중 기준 랭킹 (단일 CPC로 결정하지 않음)
# ---------------------------------------------------------------------------
def evaluate_candidate_vs_baseline(cand_metrics, base_metrics, degrade_margin=0.10):
    """
    우선순위:
      1) overall CPC가 baseline보다 개선되는가 (게이트)
      2) overall RMSE/MAE가 심각하게(>=10%) 악화되지 않는가 (안전조건)
      3) 0<OD<=100 MAE가 심각하게(>=10%) 악화되지 않는가 (안전조건)
      4) OD>=1000 RMSE가 개선되는가 (선호 신호, 게이트는 아님)
    반환: dict(passes_gate, reasons, big_od_improved, scores)
    """
    if cand_metrics is None or base_metrics is None:
        return {'passes_gate': False, 'reasons': ['metrics 없음'], 'big_od_improved': None, 'scores': {}}

    reasons = []
    c_overall = cand_metrics.get(OVERALL_NAME, {})
    b_overall = base_metrics.get(OVERALL_NAME, {})
    c_small = cand_metrics.get(SMALL_OD_BIN_NAME, {})
    b_small = base_metrics.get(SMALL_OD_BIN_NAME, {})
    c_big = cand_metrics.get(BIG_OD_BIN_NAME, {})
    b_big = base_metrics.get(BIG_OD_BIN_NAME, {})

    cpc_gate = (c_overall.get('cpc', float('-inf')) > b_overall.get('cpc', float('inf')))
    if not cpc_gate:
        reasons.append(f"overall CPC {c_overall.get('cpc')} <= baseline {b_overall.get('cpc')}")

    def _degrade_ratio(c, b):
        if b in (None, 0) or b != b:
            return None
        return (c - b) / b

    rmse_ratio = _degrade_ratio(c_overall.get('rmse'), b_overall.get('rmse'))
    mae_ratio = _degrade_ratio(c_overall.get('mae'), b_overall.get('mae'))
    small_mae_ratio = _degrade_ratio(c_small.get('mae'), b_small.get('mae'))

    safety_ok = True
    if rmse_ratio is not None and rmse_ratio >= degrade_margin:
        safety_ok = False
        reasons.append(f"overall RMSE {rmse_ratio:.1%} 악화")
    if mae_ratio is not None and mae_ratio >= degrade_margin:
        safety_ok = False
        reasons.append(f"overall MAE {mae_ratio:.1%} 악화")
    if small_mae_ratio is not None and small_mae_ratio >= degrade_margin:
        safety_ok = False
        reasons.append(f"0<OD<=100 MAE {small_mae_ratio:.1%} 악화(큰 OD 편향 의심)")

    big_od_improved = None
    if c_big.get('rmse') is not None and b_big.get('rmse') is not None:
        big_od_improved = c_big['rmse'] < b_big['rmse']

    return {
        'passes_gate': cpc_gate and safety_ok,
        'reasons': reasons,
        'big_od_improved': big_od_improved,
        'scores': {
            'overall_cpc': c_overall.get('cpc'), 'overall_rmse': c_overall.get('rmse'),
            'overall_mae': c_overall.get('mae'), 'small_od_mae': c_small.get('mae'),
            'big_od_rmse': c_big.get('rmse'),
        },
    }


def compute_pareto_frontier(entries):
    """
    entries: [(label, {'overall_cpc':..,'big_od_rmse':..,'small_od_mae':..})]
    CPC는 높을수록, RMSE/MAE는 낮을수록 좋음 — 3목적 Pareto 지배 여부를 계산.
    반환: {label: is_on_frontier(bool)}
    """
    result = {}
    for label_i, s_i in entries:
        dominated = False
        for label_j, s_j in entries:
            if label_i == label_j:
                continue
            better_or_equal = (
                s_j.get('overall_cpc', float('-inf')) >= s_i.get('overall_cpc', float('-inf')) and
                s_j.get('big_od_rmse', float('inf')) <= s_i.get('big_od_rmse', float('inf')) and
                s_j.get('small_od_mae', float('inf')) <= s_i.get('small_od_mae', float('inf'))
            )
            strictly_better = (
                s_j.get('overall_cpc', float('-inf')) > s_i.get('overall_cpc', float('-inf')) or
                s_j.get('big_od_rmse', float('inf')) < s_i.get('big_od_rmse', float('inf')) or
                s_j.get('small_od_mae', float('inf')) < s_i.get('small_od_mae', float('inf'))
            )
            if better_or_equal and strictly_better:
                dominated = True
                break
        result[label_i] = not dominated
    return result


def write_leaderboard_csv(path, metrics_by_label, meta_by_label):
    import csv
    rows = []
    for label, metrics in metrics_by_label.items():
        meta = meta_by_label.get(label, {})
        if metrics is None:
            rows.append({'label': label, **meta, 'status': 'FAILED'})
            continue
        row = {'label': label, **meta, 'status': 'OK'}
        for cat in OFFICIAL_CATEGORIES:
            m = metrics.get(cat, {})
            prefix = cat.replace(' ', '_')
            row[f'{prefix}_rmse'] = m.get('rmse')
            row[f'{prefix}_mae'] = m.get('mae')
            row[f'{prefix}_cpc'] = m.get('cpc')
            row[f'{prefix}_n'] = m.get('n_samples')
        rows.append(row)
    if not rows:
        return
    all_keys = []
    for r in rows:
        for k in r:
            if k not in all_keys:
                all_keys.append(k)
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def get_paths(out_dir):
    return {
        'out_dir': out_dir,
        'runs_dir': os.path.join(out_dir, 'runs'),
        'run_status': os.path.join(out_dir, 'RUN_STATUS.md'),
        'timing_csv': os.path.join(out_dir, 'timing.csv'),
        'failures_csv': os.path.join(out_dir, 'failures.csv'),
        'stage0_csv': os.path.join(out_dir, 'stage0_historical_reproduction.csv'),
        'screening_csv': os.path.join(out_dir, 'screening_leaderboard.csv'),
        'final_csv': os.path.join(out_dir, 'final_leaderboard.csv'),
        'derived_hparams': os.path.join(out_dir, 'derived_loss_hparams_strict.json'),
        'derived_hparams_legacy': os.path.join(out_dir, 'derived_loss_hparams_legacy.json'),
        'env_info': os.path.join(out_dir, 'env_info.json'),
        'summary': os.path.join(out_dir, 'summary.json'),
    }


def run_stage0(paths, python_exe, max_retries):
    """Historical baseline reproduction: legacy protocol, dynamic_weighted_mse, 50 epoch, seed=42."""
    append_run_status(paths['run_status'], [log("Stage 0 시작: historical baseline reproduction (legacy, dynamic_weighted_mse, 50ep)")])
    run_dir = os.path.join(paths['runs_dir'], 'stage0_historical_reproduction')
    ok, info = run_train(
        python_exe, run_dir, 'mae1', 'dynamic_weighted_mse', {}, MAIN_SEED, 'legacy',
        PLANNED_EPOCHS, stop_epoch=None, smoke_steps=None, run_final_test=True,
        max_retries=max_retries, timing_csv=paths['timing_csv'], failures_csv=paths['failures_csv'])

    result = {'ok': ok, 'run_dir': run_dir}
    if ok:
        csv_path = os.path.join(run_dir, f'results_mae1_dynamic_weighted_mse_seed{MAIN_SEED}_legacy.csv')
        metrics = load_official_metrics(csv_path)
        result['metrics'] = metrics
        if metrics:
            overall = metrics.get(OVERALL_NAME, {})
            cpc, rmse = overall.get('cpc'), overall.get('rmse')
            cpc_diff = (cpc - HISTORICAL_CPC) if cpc is not None else None
            rmse_diff = (rmse - HISTORICAL_RMSE) if rmse is not None else None
            append_run_status(paths['run_status'], [
                log(f"Stage 0 완료: 재현 CPC={cpc} (historical {HISTORICAL_CPC}, diff={cpc_diff}), "
                    f"RMSE={rmse} (historical {HISTORICAL_RMSE}, diff={rmse_diff})"),
                log("주의: seed 고정 안 됐던 과거 실행/불명확한 학습 조건 때문에 정확히 같은 값이 나올 "
                    "것으로 기대하지 않음 — 차이의 방향과 크기만 참고할 것."),
            ])
            result['cpc_diff'] = cpc_diff
            result['rmse_diff'] = rmse_diff
    else:
        append_run_status(paths['run_status'], [log("Stage 0 실패 — historical reproduction 불가, 이후 stage는 계속 진행")])
    return result


def run_stage1(paths, python_exe, derived, max_retries):
    """Runtime smoke: strict protocol, 각 후보 SMOKE_STEPS 스텝. 순위에는 사용하지 않음."""
    append_run_status(paths['run_status'], [log(f"Stage 1 시작: runtime smoke (strict, {SMOKE_STEPS} step, seed={SMOKE_SEED})")])
    passed = []
    for c in CANDIDATE_DEFS:
        params = inject_derived_params(c, derived)
        run_dir = os.path.join(paths['runs_dir'], f"stage1_smoke_{c['label']}")
        ok, info = run_train(
            python_exe, run_dir, 'mae1', c['loss'], params, SMOKE_SEED, 'strict',
            PLANNED_EPOCHS, stop_epoch=None, smoke_steps=SMOKE_STEPS, run_final_test=False,
            max_retries=max_retries, timing_csv=paths['timing_csv'], failures_csv=paths['failures_csv'])
        if ok:
            passed.append(c)
        diag = info.get('info') if isinstance(info, dict) else None
        append_run_status(paths['run_status'], [
            log(f"Stage 1 [{c['label']}] {'통과' if ok else '실패(제외)'} — 진단: {diag}")])
    append_run_status(paths['run_status'], [
        log(f"Stage 1 완료: {len(passed)}/{len(CANDIDATE_DEFS)}개 통과 ({[c['label'] for c in passed]})")])
    return passed


def run_stage2(paths, python_exe, derived, passed_stage1, max_retries):
    """Screening: strict protocol, planned 50 epoch 중 stop_epoch=15까지, seed=42."""
    append_run_status(paths['run_status'], [
        log(f"Stage 2 시작: screening (strict, planned={PLANNED_EPOCHS}ep, stop_epoch={SCREENING_STOP_EPOCH}, seed={MAIN_SEED})")])
    metrics_by_label = {}
    meta_by_label = {}
    for c in passed_stage1:
        params = inject_derived_params(c, derived)
        run_dir = os.path.join(paths['runs_dir'], f"strict_{c['label']}")  # Stage3와 공유(이어서 학습)
        ok, info = run_train(
            python_exe, run_dir, 'mae1', c['loss'], params, MAIN_SEED, 'strict',
            PLANNED_EPOCHS, stop_epoch=SCREENING_STOP_EPOCH, smoke_steps=None, run_final_test=False,
            max_retries=max_retries, timing_csv=paths['timing_csv'], failures_csv=paths['failures_csv'])
        meta_by_label[c['label']] = {'loss': c['loss'], 'params': canonical_json(params),
                                      'is_baseline': c.get('is_baseline', False)}
        if ok:
            # screening 단계에서도 validation 성능을 봐야 하므로, strict val_dataset 기준으로
            # 별도 평가를 model_test.py 대신 training_state.pt의 best checkpoint를 그대로 재사용해
            # --run-final-test 없이 이미 저장된 val 성능(RUN_STATUS의 로그)을 참고한다.
            # 여기서는 train.py가 학습 중 주기적으로 로그로 남긴 val RMSE/CPC를 다시 계산하는 대신,
            # 좀 더 신뢰성 있게 model_test.py를 val 모드로 한 번 더 돌려 공식 breakdown을 얻는다.
            metrics = run_val_evaluation(python_exe, run_dir, c, params, derived, paths)
            metrics_by_label[c['label']] = metrics
        else:
            metrics_by_label[c['label']] = None
        append_run_status(paths['run_status'], [log(f"Stage 2 [{c['label']}] {'완료' if ok else '실패'}")])

    write_leaderboard_csv(paths['screening_csv'], metrics_by_label, meta_by_label)
    return metrics_by_label, meta_by_label


def run_val_evaluation(python_exe, run_dir, candidate, params, derived, paths):
    """
    strict protocol의 validation(strict_val_indices) 기준 공식 breakdown을 얻기 위해
    model_test.py를 mode='val'로 실행한다(evaluate_breakdown을 그대로 재사용).
    """
    log_path = os.path.join(run_dir, 'val_eval.log')
    cmd = [
        python_exe, os.path.join(REPO_ROOT, 'src', 'model', 'model_test.py'),
        '--model', 'mae1', '--loss', candidate['loss'], '--loss-params', json.dumps(params),
        '--seed', str(MAIN_SEED), '--protocol', 'strict', '--epochs', str(PLANNED_EPOCHS), '--batch_size', '32',
    ]
    with open(log_path, 'w', encoding='utf-8') as logf:
        subprocess.run(cmd, cwd=run_dir, stdout=logf, stderr=subprocess.STDOUT)
    csv_path = os.path.join(run_dir, f"results_mae1_{candidate['loss']}_seed{MAIN_SEED}_strict.csv")
    return load_official_metrics(csv_path)


def run_stage3(paths, python_exe, derived, screening_metrics, screening_meta, max_retries, time_budget_hours):
    """Final: strict protocol, baseline + screening 상위 1~2개, 50 epoch(전체), 최종 test는 여기서만."""
    baseline_label = next(c['label'] for c in CANDIDATE_DEFS if c.get('is_baseline'))
    baseline_metrics = screening_metrics.get(baseline_label)

    ranked = []
    for c in CANDIDATE_DEFS:
        if c.get('is_baseline') or c['label'] not in screening_metrics:
            continue
        m = screening_metrics[c['label']]
        verdict = evaluate_candidate_vs_baseline(m, baseline_metrics)
        ranked.append((c, verdict))

    ranked_passing = [(c, v) for c, v in ranked if v['passes_gate']]
    ranked_passing.sort(key=lambda cv: (cv[1]['scores'].get('overall_cpc') or float('-inf')), reverse=True)

    top_n = 2
    final_new = ranked_passing[:top_n]

    append_run_status(paths['run_status'], [
        log(f"Stage 3 후보 선정: {[c['label'] for c, v in final_new]} "
            f"(gate 통과 {len(ranked_passing)}/{len(ranked)}개 중 상위 {top_n})"),
    ])

    final_entries = [next(c for c in CANDIDATE_DEFS if c.get('is_baseline'))] + [c for c, v in final_new]

    # 시간 예산 확인(screening의 timing.csv에서 epoch당 시간 추정)
    per_epoch_time = _estimate_epoch_time(paths['timing_csv'])
    if per_epoch_time:
        projected_hours = len(final_entries) * PLANNED_EPOCHS * per_epoch_time / 3600.0
        append_run_status(paths['run_status'], [
            log(f"Stage 3 예상 시간: 약 {projected_hours:.2f}시간 ({len(final_entries)}개 x {PLANNED_EPOCHS}ep, "
                f"epoch당 {per_epoch_time:.1f}s 기준)")])
        if projected_hours > time_budget_hours and len(final_entries) > 2:
            final_entries = final_entries[:2]
            append_run_status(paths['run_status'], [log(f"시간 예산({time_budget_hours}h) 초과 예상 -> 상위 1개만 진행")])

    metrics_by_label = {}
    meta_by_label = {}
    for c in final_entries:
        params = inject_derived_params(c, derived)
        run_dir = os.path.join(paths['runs_dir'], f"strict_{c['label']}")  # Stage2와 동일 디렉토리(이어서 학습)
        ok, info = run_train(
            python_exe, run_dir, 'mae1', c['loss'], params, MAIN_SEED, 'strict',
            PLANNED_EPOCHS, stop_epoch=None, smoke_steps=None, run_final_test=True,
            max_retries=max_retries, timing_csv=paths['timing_csv'], failures_csv=paths['failures_csv'])
        meta_by_label[c['label']] = {'loss': c['loss'], 'params': canonical_json(params),
                                      'is_baseline': c.get('is_baseline', False)}
        if ok:
            csv_path = os.path.join(run_dir, f"results_mae1_{c['loss']}_seed{MAIN_SEED}_strict.csv")
            metrics_by_label[c['label']] = load_official_metrics(csv_path)
        else:
            metrics_by_label[c['label']] = None
        append_run_status(paths['run_status'], [log(f"Stage 3 [{c['label']}] {'완료' if ok else '실패'} — 다음 실험 계속 진행")])

    write_leaderboard_csv(paths['final_csv'], metrics_by_label, meta_by_label)
    return metrics_by_label, meta_by_label, ranked


def _estimate_epoch_time(timing_csv_path):
    if not os.path.exists(timing_csv_path):
        return None
    import csv
    times = []
    with open(timing_csv_path, 'r', encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            if row.get('success') == 'True':
                try:
                    wt = float(row['wall_time_sec'])
                    # timing.csv에는 total wall time만 있으므로 대략 15epoch 기준으로 나눈다(screening epoch 수)
                    times.append(wt / SCREENING_STOP_EPOCH)
                except (ValueError, KeyError):
                    pass
    return sum(times) / len(times) if times else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--python', type=str, required=True)
    parser.add_argument('--stage', type=str, default='all', choices=['0', '1', '2', '3', 'report', 'all'])
    parser.add_argument('--max-retries', type=int, default=2)
    parser.add_argument('--time-budget-hours', type=float, default=8.0)
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    paths = get_paths(out_dir)
    os.makedirs(paths['runs_dir'], exist_ok=True)

    if not os.path.exists(paths['env_info']):
        env_info = capture_env_info(args.python)
        with open(paths['env_info'], 'w', encoding='utf-8') as f:
            json.dump(env_info, f, ensure_ascii=False, indent=2)
        append_run_status(paths['run_status'], [log(f"환경 정보: {env_info}")])

    if not os.path.exists(paths['derived_hparams']):
        append_run_status(paths['run_status'], [
            log("치명적 오류: derived_loss_hparams_strict.json이 없어 실행 중단 "
                "(먼저 scripts/analyze_target_distribution.py --protocol strict --out-dir ... 실행 필요, "
                "결과를 derived_loss_hparams_strict.json으로 이 out-dir에 저장할 것)")])
        sys.exit(1)
    with open(paths['derived_hparams'], 'r', encoding='utf-8') as f:
        derived = json.load(f)

    t_start = time.time()

    stage0_result = None
    if args.stage in ('0', 'all'):
        stage0_result = run_stage0(paths, args.python, args.max_retries)

    passed_stage1 = None
    if args.stage in ('1', 'all'):
        passed_stage1 = run_stage1(paths, args.python, derived, args.max_retries)
    elif args.stage in ('2', '3'):
        # 재개 시 CANDIDATE_DEFS 전체를 다시 smoke 통과한 것으로 간주(스킵 로직이 개별 판단하므로 안전)
        passed_stage1 = CANDIDATE_DEFS

    screening_metrics, screening_meta = None, None
    if args.stage in ('2', 'all'):
        screening_metrics, screening_meta = run_stage2(paths, args.python, derived, passed_stage1 or CANDIDATE_DEFS, args.max_retries)
    elif args.stage in ('3', 'report'):
        if os.path.exists(paths['screening_csv']):
            screening_metrics, screening_meta = _read_leaderboard(paths['screening_csv'])

    final_metrics, final_meta, ranked = None, None, None
    if args.stage in ('3', 'all'):
        if screening_metrics is None:
            append_run_status(paths['run_status'], [log("Stage 3 중단: screening 결과가 없습니다. 먼저 --stage 2를 실행하세요.")])
        else:
            final_metrics, final_meta, ranked = run_stage3(
                paths, args.python, derived, screening_metrics, screening_meta, args.max_retries, args.time_budget_hours)

    if args.stage in ('report', 'all'):
        write_final_report(paths, stage0_result, passed_stage1, screening_metrics, screening_meta,
                            final_metrics, final_meta, args.python)

    elapsed_h = (time.time() - t_start) / 3600.0
    append_run_status(paths['run_status'], [log(f"[--stage {args.stage}] 이번 호출 소요 시간: {elapsed_h:.2f}시간")])


def _read_leaderboard(path):
    import csv
    metrics_by_label, meta_by_label = {}, {}
    if not os.path.exists(path):
        return {}, {}
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            label = row['label']
            meta_by_label[label] = {'loss': row.get('loss'), 'params': row.get('params'),
                                     'is_baseline': row.get('is_baseline') == 'True'}
            if row.get('status') == 'FAILED':
                metrics_by_label[label] = None
                continue
            metrics = {}
            for cat in OFFICIAL_CATEGORIES:
                prefix = cat.replace(' ', '_')
                if f'{prefix}_rmse' not in row:
                    continue

                def _f(v):
                    try:
                        return float(v) if v not in ('', None) else float('nan')
                    except ValueError:
                        return float('nan')
                metrics[cat] = {'rmse': _f(row.get(f'{prefix}_rmse')), 'mae': _f(row.get(f'{prefix}_mae')),
                                 'cpc': _f(row.get(f'{prefix}_cpc')),
                                 'n_samples': int(float(row[f'{prefix}_n'])) if row.get(f'{prefix}_n') not in ('', None) else 0}
            metrics_by_label[label] = metrics if metrics else None
    return metrics_by_label, meta_by_label


def write_final_report(paths, stage0_result, passed_stage1, screening_metrics, screening_meta,
                        final_metrics, final_meta, python_exe):
    lines = []
    a = lines.append
    env_info = {}
    if os.path.exists(paths['env_info']):
        with open(paths['env_info'], 'r', encoding='utf-8') as f:
            env_info = json.load(f)
    git_commit = get_git_commit_hash()

    a("# MAE1(SpatialODMAE1) Loss 탐색 최종 리포트 (v2)")
    a("")
    a("## 1. 실행 환경")
    a(f"- torch: {env_info.get('torch_version')}, cuda_available: {env_info.get('cuda_available')}, "
      f"cuda_device: {env_info.get('cuda_device_name')}, mps_available: {env_info.get('mps_available')}")
    a(f"- platform: {env_info.get('platform')}")
    a("")
    a("## 2. Git commit hash")
    a(f"- `{git_commit}`")
    a("")
    a("## 3. 사용 데이터와 평가 대상")
    a("- dataset/od_data.csv, dist_data.csv, final_static_features.csv, OD_dong_list.xlsx (변경 없음)")
    a("- 공식 평가 구간: " + ", ".join(OFFICIAL_CATEGORIES))
    a("")
    a("## 4. Stage 0: Historical baseline reproduction")
    if stage0_result:
        a(f"- 성공 여부: {stage0_result.get('ok')}")
        if stage0_result.get('metrics'):
            overall = stage0_result['metrics'].get(OVERALL_NAME, {})
            a(f"- 재현 CPC: {overall.get('cpc')} (historical: {HISTORICAL_CPC}, diff: {stage0_result.get('cpc_diff')})")
            a(f"- 재현 RMSE: {overall.get('rmse')} (historical: {HISTORICAL_RMSE}, diff: {stage0_result.get('rmse_diff')})")
            a("- 주의: legacy protocol은 test 지역이 checkpoint 선택에도 사용된 leakage가 있는 재현 전용 "
              "프로토콜이며, 엄밀한 untouched test 성능이 아님.")
    else:
        a("- 실행되지 않음(--stage 0 미실행)")
    a("")
    a("## 5. Stage 1 결과(smoke)")
    a(f"- 통과 후보: {[c['label'] for c in passed_stage1] if passed_stage1 else '실행 안 됨'}")
    a("")
    a("## 6~9. Screening/Final leaderboard, 추천")
    baseline_label = next(c['label'] for c in CANDIDATE_DEFS if c.get('is_baseline'))
    if final_metrics:
        a("### Final(50 epoch) leaderboard")
        a("| label | overall CPC | overall RMSE | overall MAE | 0<OD<=100 MAE | OD>=1000 RMSE |")
        a("|---|---|---|---|---|---|")
        for label, m in final_metrics.items():
            if m is None:
                a(f"| {label} | FAILED | | | | |")
                continue
            o, s, b = m.get(OVERALL_NAME, {}), m.get(SMALL_OD_BIN_NAME, {}), m.get(BIG_OD_BIN_NAME, {})
            a(f"| {label} | {o.get('cpc')} | {o.get('rmse')} | {o.get('mae')} | {s.get('mae')} | {b.get('rmse')} |")

        base_m = final_metrics.get(baseline_label)
        entries_for_pareto = []
        recommendation = None
        best_cpc = float('-inf')
        for label, m in final_metrics.items():
            if label == baseline_label or m is None:
                continue
            verdict = evaluate_candidate_vs_baseline(m, base_m)
            entries_for_pareto.append((label, verdict['scores']))
            if verdict['passes_gate'] and (verdict['scores'].get('overall_cpc') or float('-inf')) > best_cpc:
                best_cpc = verdict['scores']['overall_cpc']
                recommendation = label

        pareto = compute_pareto_frontier(entries_for_pareto) if entries_for_pareto else {}
        a("")
        a(f"### Pareto frontier (overall_cpc/big_od_rmse/small_od_mae 기준): {pareto}")
        a("")
        a("## 15. 추천 loss")
        if recommendation:
            a(f"- **{recommendation}**")
        else:
            a(f"- 추천 없음 — 모든 신규 loss가 baseline(**{baseline_label}**)보다 개선되지 못했거나 안전조건에서 "
              f"제외됨. **{baseline_label}**(dynamic_weighted_mse) 유지를 권장.")
    else:
        a("- Stage 3(final)가 실행되지 않아 최종 leaderboard 없음")
    a("")
    a("## 18. 단일 seed 실험이라는 한계")
    a(f"- 모든 정식 실험(Stage 0/2/3)은 seed={MAIN_SEED} 단일 시드로 수행됨(Stage 1 smoke만 seed={SMOKE_SEED}). "
      "seed-to-seed 분산을 고려하지 않았으므로 순위는 하나의 표본에 불과함.")
    a("")
    a("## 19. 다음 단계")
    a("- 추천 loss에 대한 다중 seed(7, 2026 등) 재현성 검증")
    a("- 제외된 후보(안전조건 위반)의 하이퍼파라미터 재조정")
    a("- Tweedie(모델 출력층 재설계 필요), Huber/PINN 계열(이미 실패로 분류)은 이번 범위 밖")
    a("")

    with open(os.path.join(paths['out_dir'], 'final_report.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    append_run_status(paths['run_status'], [log("final_report.md 작성 완료")])


if __name__ == '__main__':
    main()
