#!/usr/bin/env python3
"""
SpatialODMAE1(mae1) loss 탐색 자동 실험 오케스트레이터.

Stage 1: weighted_mse baseline smoke test (2 epoch, seed=999)
Stage 2: 신규 loss 5종 x 하이퍼파라미터 조합 smoke test (2 epoch, seed=999)
Stage 3: baseline + Stage2 통과 후보 전체 10 epoch screening (seed=42)
Stage 4: baseline + screening 상위 후보 50 epoch 최종 실험 (seed=42)

이 스크립트가 직접 건드리는 것은 "어떤 loss/하이퍼파라미터로 몇 epoch 학습할지"뿐이며,
실제 학습/평가 로직(src/model/train.py, src/model/model_test.py, src/model/models.py,
src/model/dataset.py, src/eval_utils.py)은 subprocess로 그대로 호출만 한다 — 모델 구조,
데이터, masking, optimizer, lr, scheduler, epochs/batch_size(각 stage 내에서는 고정),
seed, checkpoint 선정 기준, 평가 코드는 모두 동일하게 유지된다.
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PY = os.path.join(REPO_ROOT, 'src', 'model', 'train.py')
ANALYZE_PY = os.path.join(REPO_ROOT, 'scripts', 'analyze_target_distribution.py')

OFFICIAL_CATEGORIES = [
    '전체 마스킹 평가 대상', '실제값 0', '실제값 1~100', '실제값 101~999',
    '실제값 1000이상', '동일 행정동 내부 이동', '서로 다른 행정동 간 이동',
]

# ---------------------------------------------------------------------------
# 신규 loss 후보 정의. params 안의 global_scale/bin_freq/tau는 실행 시
# derived_loss_hparams.json 값으로 덮어씀(데이터 기반 값 사용).
# ---------------------------------------------------------------------------
def build_candidate_defs():
    return [
        {'loss': 'dual_scale_mse', 'label': 'dual_scale_mse_lam0.5', 'params': {'lambda_log': 0.5}},
        {'loss': 'dual_scale_mse', 'label': 'dual_scale_mse_lam0.7', 'params': {'lambda_log': 0.7}},
        {'loss': 'bin_balanced_mse', 'label': 'bin_balanced_mse_invfreq', 'params': {'weight_mode': 'inv_freq'}},
        {'loss': 'bin_balanced_mse', 'label': 'bin_balanced_mse_invsqrt', 'params': {'weight_mode': 'inv_sqrt_freq'}},
        {'loss': 'cpc_hybrid', 'label': 'cpc_hybrid_lam0.1', 'params': {'lambda_cpc': 0.1}},
        {'loss': 'cpc_hybrid', 'label': 'cpc_hybrid_lam0.3', 'params': {'lambda_cpc': 0.3}},
        {'loss': 'tweedie_deviance', 'label': 'tweedie_p1.3', 'params': {'p': 1.3}},
        {'loss': 'tweedie_deviance', 'label': 'tweedie_p1.5', 'params': {'p': 1.5}},
        {'loss': 'tweedie_deviance', 'label': 'tweedie_p1.7', 'params': {'p': 1.7}},
        {'loss': 'tail_aware_relative', 'label': 'tail_aware_relative_a', 'params': {'lambda_log': 0.7, 'lambda_relative': 0.3}},
        {'loss': 'tail_aware_relative', 'label': 'tail_aware_relative_b', 'params': {'lambda_log': 0.5, 'lambda_relative': 0.5}},
    ]


def inject_derived_params(candidate, derived):
    params = dict(candidate['params'])
    if candidate['loss'] == 'dual_scale_mse':
        params['global_scale'] = derived['dual_scale_mse']['global_scale']
    elif candidate['loss'] == 'bin_balanced_mse':
        params['bin_freq'] = derived['bin_balanced_mse']['bin_freq']
    elif candidate['loss'] == 'tail_aware_relative':
        params['tau'] = derived['tail_aware_relative']['tau']
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


def run_train_subprocess(python_exe, run_dir, model, epochs, batch_size, loss, loss_params, seed, log_path):
    os.makedirs(run_dir, exist_ok=True)
    cmd = [
        python_exe, TRAIN_PY,
        '--model', model,
        '--epochs', str(epochs),
        '--batch_size', str(batch_size),
        '--loss', loss,
        '--loss-params', json.dumps(loss_params),
        '--seed', str(seed),
    ]
    start = time.time()
    with open(log_path, 'a', encoding='utf-8') as logf:
        logf.write(f"\n=== CMD: {' '.join(cmd)} ===\n")
        logf.write(f"=== cwd: {run_dir} ===\n")
        logf.flush()
        proc = subprocess.run(cmd, cwd=run_dir, stdout=logf, stderr=subprocess.STDOUT)
    elapsed = time.time() - start
    return proc.returncode, elapsed


def scan_log_for_instability(log_path):
    """train.log에서 traceback / nan / inf 징후를 찾는다. (True=문제 있음, 메시지)"""
    if not os.path.exists(log_path):
        return True, 'log 파일이 생성되지 않음'
    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()
    if 'Traceback (most recent call last)' in text:
        tail = text.strip().splitlines()[-15:]
        return True, 'Python traceback 발견: ' + ' | '.join(tail)
    import re
    # tqdm postfix 등에서 'loss': nan / inf 형태로 찍히는 걸 탐지
    if re.search(r"\bnan\b", text, re.IGNORECASE) or re.search(r"[^a-zA-Z]inf[^a-zA-Z]", text):
        return True, 'nan/inf 문자열이 로그에서 발견됨'
    return False, None


def find_result_files(run_dir, model, loss, seed):
    ckpt = os.path.join(run_dir, f'best_model_{model}_{loss}_seed{seed}.pth')
    csv_path = os.path.join(run_dir, f'results_{model}_{loss}_seed{seed}.csv')
    png_path = os.path.join(run_dir, f'results_{model}_{loss}_seed{seed}.png')
    return ckpt, csv_path, png_path


def normalize_run_artifacts(run_dir, ckpt, csv_path, png_path, log_path):
    """요청된 표준 파일명(results.csv/visualization.png/best_model.pth/train.log)으로 복사본을 남긴다."""
    mapping = [
        (csv_path, os.path.join(run_dir, 'results.csv')),
        (png_path, os.path.join(run_dir, 'visualization.png')),
        (ckpt, os.path.join(run_dir, 'best_model.pth')),
    ]
    for src, dst in mapping:
        if os.path.exists(src) and not os.path.exists(dst):
            try:
                shutil.copy2(src, dst)
            except Exception:
                pass
    # train.log는 이미 표준 이름으로 기록 중


TIMING_FIELDS = ['run_name', 'stage', 'loss', 'params', 'seed', 'epochs', 'attempt',
                  'wall_time_sec', 'epoch_time_sec_avg', 'returncode', 'success']
FAILURE_FIELDS = ['run_name', 'stage', 'loss', 'params', 'attempt', 'error_summary', 'log_path']


def append_csv_row(path, fields, row):
    """
    파일이 없으면 헤더부터 써서 새로 만들고, 있으면 append만 한다.
    Colab에서 stage별로 별도 셀(=별도 프로세스)로 나눠 실행해도 이전 stage가 쓴 내용이
    사라지지 않도록 "메모리에 누적 후 한번에 덮어쓰기" 대신 매 row마다 즉시 append한다.
    """
    is_new = not os.path.exists(path)
    with open(path, 'a', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def read_csv_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def read_leaderboard_csv(path):
    """write_leaderboard_csv가 쓴 wide-format CSV를 (metrics_by_run, run_meta_by_run)으로 복원."""
    metrics_by_run = {}
    run_meta_by_run = {}
    for row in read_csv_rows(path):
        run_name = row['run_name']
        run_meta_by_run[run_name] = {'loss': row.get('loss'), 'label': row.get('label'),
                                      'params': row.get('params')}
        if row.get('status') == 'FAILED':
            metrics_by_run[run_name] = None
            continue
        metrics = {}
        for cat in OFFICIAL_CATEGORIES:
            prefix = cat.replace(' ', '_')
            rmse_key, mae_key, cpc_key, n_key = f'{prefix}_rmse', f'{prefix}_mae', f'{prefix}_cpc', f'{prefix}_n'
            if rmse_key not in row:
                continue
            def _f(v):
                try:
                    return float(v) if v not in ('', None) else float('nan')
                except ValueError:
                    return float('nan')
            metrics[cat] = {
                'rmse': _f(row.get(rmse_key)), 'mae': _f(row.get(mae_key)), 'cpc': _f(row.get(cpc_key)),
                'n_samples': int(float(row[n_key])) if row.get(n_key) not in ('', None) else 0,
            }
        metrics_by_run[run_name] = metrics if metrics else None
    return metrics_by_run, run_meta_by_run


def load_official_metrics(csv_path):
    """results CSV에서 evaluation_type=official_masked 행만 category별로 dict화."""
    if not os.path.exists(csv_path):
        return None
    metrics = {}
    with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('evaluation_type') != 'official_masked':
                continue
            cat = row.get('category')
            try:
                metrics[cat] = {
                    'rmse': float(row['rmse']) if row['rmse'] not in ('', 'nan') else float('nan'),
                    'mae': float(row['mae']) if row['mae'] not in ('', 'nan') else float('nan'),
                    'cpc': float(row['cpc']) if row['cpc'] not in ('', 'nan') else float('nan'),
                    'n_samples': int(float(row['n_samples'])) if row['n_samples'] not in ('',) else 0,
                }
            except (KeyError, ValueError):
                continue
    return metrics if metrics else None


def run_one(python_exe, base_run_dir, stage_name, run_name, model, epochs, batch_size,
            loss, loss_params, seed, max_retries, timing_csv_path, failures_csv_path):
    run_dir = os.path.join(base_run_dir, run_name)
    log_path = os.path.join(run_dir, 'train.log')
    os.makedirs(run_dir, exist_ok=True)

    # --- skip-if-already-done: Colab 세션이 끊겼다 재개되는 경우를 위한 idempotent 실행 ---
    ckpt, csv_path, png_path = find_result_files(run_dir, model, loss, seed)
    if os.path.exists(ckpt) and os.path.exists(csv_path):
        log(f"[{stage_name}] {run_name} 이미 완료된 결과 발견 — 재실행 없이 재사용")
        return True, run_dir, csv_path

    attempt = 0
    while True:
        attempt += 1
        log(f"[{stage_name}] {run_name} 시작 (attempt {attempt}/{max_retries + 1}) "
            f"loss={loss} params={loss_params} epochs={epochs} seed={seed}")
        try:
            returncode, elapsed = run_train_subprocess(
                python_exe, run_dir, model, epochs, batch_size, loss, loss_params, seed, log_path)
        except Exception as e:
            returncode = -1
            elapsed = 0.0
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(f"\n=== 오케스트레이터 예외: {e}\n{traceback.format_exc()}\n")

        unstable, reason = scan_log_for_instability(log_path)
        ckpt_ok = os.path.exists(ckpt)
        csv_ok = os.path.exists(csv_path)

        success = (returncode == 0) and (not unstable) and ckpt_ok and csv_ok

        append_csv_row(timing_csv_path, TIMING_FIELDS, {
            'run_name': run_name, 'stage': stage_name, 'loss': loss,
            'params': json.dumps(loss_params, ensure_ascii=False), 'seed': seed, 'epochs': epochs,
            'attempt': attempt, 'wall_time_sec': round(elapsed, 2),
            'epoch_time_sec_avg': round(elapsed / epochs, 2) if epochs else None,
            'returncode': returncode, 'success': success,
        })

        if success:
            normalize_run_artifacts(run_dir, ckpt, csv_path, png_path, log_path)
            log(f"[{stage_name}] {run_name} 성공 ({elapsed:.1f}s)")
            return True, run_dir, csv_path
        else:
            error_summary = reason or f'returncode={returncode}, ckpt_ok={ckpt_ok}, csv_ok={csv_ok}'
            append_csv_row(failures_csv_path, FAILURE_FIELDS, {
                'run_name': run_name, 'stage': stage_name, 'loss': loss,
                'params': json.dumps(loss_params, ensure_ascii=False), 'attempt': attempt,
                'error_summary': error_summary, 'log_path': log_path,
            })
            log(f"[{stage_name}] {run_name} 실패 (attempt {attempt}): {error_summary}")
            if attempt > max_retries:
                log(f"[{stage_name}] {run_name} 최종 실패 — 이후 stage에서 제외하고 계속 진행")
                return False, run_dir, None


def rank_candidates(metrics_by_run, baseline_run_name, safety_margin=0.10, cpc_tie_eps=0.01):
    """
    metrics_by_run: {run_name: {category: {rmse, mae, cpc, n_samples}}}
    반환: (ranked_list[(run_name, score_dict)], excluded_large_od_bias[run_name])
    1차: overall masked CPC 높은 순
    2차: CPC 차이가 0.01 이내면 OD>=1000 RMSE 낮은 순
    안전조건: OD 1~100 MAE가 baseline 대비 10% 이상 악화되면 "큰 OD 편향"으로 제외
    """
    baseline_metrics = metrics_by_run.get(baseline_run_name)
    baseline_mae_small = None
    if baseline_metrics and '실제값 1~100' in baseline_metrics:
        baseline_mae_small = baseline_metrics['실제값 1~100']['mae']

    candidates = []
    excluded = {}
    for run_name, metrics in metrics_by_run.items():
        if metrics is None or '전체 마스킹 평가 대상' not in metrics:
            excluded[run_name] = '공식 지표(overall) 없음'
            continue
        overall = metrics['전체 마스킹 평가 대상']
        big_od = metrics.get('실제값 1000이상', {})
        small_od = metrics.get('실제값 1~100', {})

        if run_name != baseline_run_name and baseline_mae_small is not None and small_od:
            small_mae = small_od.get('mae', float('nan'))
            if small_mae == small_mae and baseline_mae_small == baseline_mae_small and baseline_mae_small > 0:
                degrade_ratio = (small_mae - baseline_mae_small) / baseline_mae_small
                if degrade_ratio >= safety_margin:
                    excluded[run_name] = (f'큰 OD 편향(작은 OD MAE {degrade_ratio:.1%} 악화, '
                                           f'baseline={baseline_mae_small:.3f} -> {small_mae:.3f})')
                    continue

        candidates.append((run_name, {
            'overall_cpc': overall.get('cpc', float('nan')),
            'overall_rmse': overall.get('rmse', float('nan')),
            'big_od_rmse': big_od.get('rmse', float('nan')),
        }))

    def sort_key(item):
        _, s = item
        cpc = s['overall_cpc'] if s['overall_cpc'] == s['overall_cpc'] else -1.0
        return cpc

    candidates.sort(key=sort_key, reverse=True)

    # 2차 기준: 상위권 내에서 CPC 차이가 cpc_tie_eps 이내면 big_od_rmse로 재정렬
    if candidates:
        top_cpc = candidates[0][1]['overall_cpc']
        tie_group = [c for c in candidates if (top_cpc - c[1]['overall_cpc']) <= cpc_tie_eps]
        rest = [c for c in candidates if c not in tie_group]
        tie_group.sort(key=lambda c: (c[1]['big_od_rmse'] if c[1]['big_od_rmse'] == c[1]['big_od_rmse'] else float('inf')))
        candidates = tie_group + rest

    return candidates, excluded


def write_leaderboard_csv(path, metrics_by_run, run_meta_by_run):
    rows = []
    for run_name, metrics in metrics_by_run.items():
        meta = run_meta_by_run.get(run_name, {})
        if metrics is None:
            rows.append({'run_name': run_name, **meta, 'status': 'FAILED'})
            continue
        row = {'run_name': run_name, **meta, 'status': 'OK'}
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
        for k in r.keys():
            if k not in all_keys:
                all_keys.append(k)
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


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
            "'platform': platform.platform(),"
            "'machine': platform.machine()}))"
        ]).decode().strip()
        info.update(json.loads(out))
    except Exception as e:
        info['error'] = str(e)
    return info


def fmt(v, nd=4):
    if v is None:
        return 'N/A'
    try:
        if v != v:  # NaN
            return 'NaN'
        return f'{v:.{nd}f}'
    except Exception:
        return str(v)


def write_final_report(out_dir, experiment_config, env_info, candidate_defs,
                        passed_stage2, screening_metrics, screening_run_meta,
                        final_metrics, final_run_meta, excluded_stage3, excluded_stage4,
                        ranked_final_new_only, baseline_run_name_final,
                        timing_rows, failure_rows, total_elapsed_hours):
    lines = []
    a = lines.append

    a("# MAE1(SpatialODMAE1) Loss 탐색 최종 리포트")
    a("")
    a("## 1. 실행 환경과 MPS 정보")
    a(f"- python: `{env_info.get('python_exe')}`")
    a(f"- torch: {env_info.get('torch_version')}")
    a(f"- mps.is_built(): {env_info.get('mps_built')} / mps.is_available(): {env_info.get('mps_available')} / cuda.is_available(): {env_info.get('cuda_available')}")
    a(f"- platform: {env_info.get('platform')} ({env_info.get('machine')})")
    a("")
    a("## 2. Git commit hash")
    a(f"- `{experiment_config.get('git_commit')}`")
    a("")
    a("## 3. 사용 데이터와 평가 대상")
    a("- `dataset/od_data.csv`, `dataset/dist_data.csv`, `dataset/final_static_features.csv`, `dataset/raw/OD_dong_list.xlsx` (기존 코드 그대로, 변경 없음)")
    a("- 공식 평가: 테스트 지역(동탄/위례/검단) 마스킹 OD만 대상(`evaluation_type=official_masked`), 전체 N×N 결과는 `diagnostic_full_matrix`로 참고용 분리")
    a("")
    a("## 4. 실제 총 실행 시간")
    a(f"- 총 {total_elapsed_hours:.2f}시간 (Stage 1~4 전체)")
    a("")
    a("## 5. 기존에 이미 시도된 loss와 이번 신규 loss 구분")
    a("- 기존/반복 대상 아님(baseline으로만 사용): Pure MSE, MAE/L1, Huber, 기존 Weighted MSE, heavy-tail Weighted MSE 변형, Huber+PINN, row/column total constraint — 이번 실험에서 신규 후보로 다루지 않음")
    a("- 신규 후보(이번 실험 대상): dual_scale_mse, bin_balanced_mse, cpc_hybrid, tweedie_deviance, tail_aware_relative")
    a("")
    a("## 6. 신규 loss별 정확한 수식")
    a("`src/model/loss.py` 참고(각 클래스 docstring에 수식·설계 근거 전체 명시). 요약:")
    a("- **DualScaleMSELoss**: `L = lambda_log*MSE(pred_log,target_log) + (1-lambda_log)*MSE(pred_real,target_real)/detached_scale`")
    a("- **BinBalancedMSELoss**: 구간별 inverse-frequency(또는 inverse-sqrt-frequency) 가중치를 적용한 log1p MSE, 배치 평균 가중치 1로 정규화 + 상한 clamp")
    a("- **CPCHybridLoss**: `L = MSE(pred_log,target_log) + lambda_cpc*(1 - SoftCPC)`, `SoftCPC`는 `min(a,b)=(a+b-|a-b|)/2` 항등식 기반")
    a("- **TweedieDevianceLoss**: `1<p<2` Tweedie deviance, `mu=clamp(expm1(pred_log),min=eps)`로 출력층 변경 없이 양수화")
    a("- **TailAwareRelativeLoss**: `L = lambda_log*MSE(pred_log,target_log) + lambda_relative*mean(relative_error^2)`, `target_real==0`은 상대오차 항에서 제외")
    a("")
    a("## 7. 하이퍼파라미터 선정 이유")
    a("- `target_distribution.md`/`derived_loss_hparams.json`의 학습 데이터 전역 통계를 근거로 사용:")
    src = experiment_config.get('derived_loss_hparams', {}).get('source', {})
    a(f"  - 0 비율 {fmt(src.get('zero_ratio'), 4)}, 양수 median {fmt(src.get('positive_median'), 3)}, 전역 RMS {fmt(src.get('global_scale_rms_all'), 2)}")
    a(f"  - `dual_scale_mse.global_scale` = 전역 RMS({fmt(experiment_config.get('derived_loss_hparams',{}).get('dual_scale_mse',{}).get('global_scale'),2)}), 배치별 detached RMS와 동일한 정의를 데이터 전체로 고정")
    a(f"  - `bin_balanced_mse.bin_freq` = 실측 구간 비율({experiment_config.get('derived_loss_hparams',{}).get('bin_balanced_mse',{}).get('bin_freq')})")
    a(f"  - `tail_aware_relative.tau` = 양수 OD의 median({fmt(experiment_config.get('derived_loss_hparams',{}).get('tail_aware_relative',{}).get('tau'),3)})")
    a("  - `dual_scale_mse.lambda_log` ∈ {0.5, 0.7}, `cpc_hybrid.lambda_cpc` ∈ {0.1, 0.3}, `tweedie_deviance.p` ∈ {1.3, 1.5, 1.7}, "
      "`bin_balanced_mse.weight_mode` ∈ {inv_freq, inv_sqrt_freq}, `tail_aware_relative`는 (lambda_log,lambda_relative) ∈ {(0.7,0.3),(0.5,0.5)} 를 screening")
    a("")
    a("## 8. smoke test 및 screening 결과")
    a(f"- Stage 2(smoke test) 통과: {len(passed_stage2)}/{len(candidate_defs)} — {passed_stage2}")
    failed_stage2 = [c['label'] for c in candidate_defs if c['label'] not in passed_stage2]
    a(f"- Stage 2 실패(제외): {failed_stage2 if failed_stage2 else '없음'}")
    a("")
    a("### Screening(10 epoch) leaderboard — 공식(official_masked) 지표")
    a("| run | loss | overall CPC | overall RMSE | OD1000+ RMSE | OD1~100 MAE |")
    a("|---|---|---|---|---|---|")
    for run_name, metrics in screening_metrics.items():
        meta = screening_run_meta.get(run_name, {})
        if metrics is None:
            a(f"| {run_name} | {meta.get('loss')} | FAILED | | | |")
            continue
        overall = metrics.get('전체 마스킹 평가 대상', {})
        big = metrics.get('실제값 1000이상', {})
        small = metrics.get('실제값 1~100', {})
        a(f"| {run_name} | {meta.get('loss')} | {fmt(overall.get('cpc'))} | {fmt(overall.get('rmse'),2)} | {fmt(big.get('rmse'),2)} | {fmt(small.get('mae'),3)} |")
    a(f"- Stage 3 안전조건(작은 OD MAE 10%+ 악화) 제외: {excluded_stage3 if excluded_stage3 else '없음'}")
    a("")
    a("## 9. 최종 50 epoch leaderboard")
    a("| run | loss | overall CPC | overall RMSE | overall MAE | OD1000+ RMSE/MAE/CPC | OD1~100 RMSE/MAE/CPC | 실제값0 CPC | 동일동 CPC | 동간 CPC |")
    a("|---|---|---|---|---|---|---|---|---|---|")
    for run_name, metrics in final_metrics.items():
        meta = final_run_meta.get(run_name, {})
        if metrics is None:
            a(f"| {run_name} | {meta.get('loss')} | FAILED | | | | | | | |")
            continue
        overall = metrics.get('전체 마스킹 평가 대상', {})
        big = metrics.get('실제값 1000이상', {})
        small = metrics.get('실제값 1~100', {})
        zero = metrics.get('실제값 0', {})
        same = metrics.get('동일 행정동 내부 이동', {})
        diff = metrics.get('서로 다른 행정동 간 이동', {})
        a(f"| {run_name} | {meta.get('loss')} | {fmt(overall.get('cpc'))} | {fmt(overall.get('rmse'),2)} | {fmt(overall.get('mae'),2)} | "
          f"{fmt(big.get('rmse'),2)}/{fmt(big.get('mae'),2)}/{fmt(big.get('cpc'))} | "
          f"{fmt(small.get('rmse'),2)}/{fmt(small.get('mae'),3)}/{fmt(small.get('cpc'))} | "
          f"{fmt(zero.get('cpc'))} | {fmt(same.get('cpc'))} | {fmt(diff.get('cpc'))} |")
    a("")
    a("## 10. Weighted MSE 대비 변화량과 변화율")
    baseline_metrics = final_metrics.get(baseline_run_name_final)
    if baseline_metrics:
        b_overall = baseline_metrics.get('전체 마스킹 평가 대상', {})
        for run_name, metrics in final_metrics.items():
            if run_name == baseline_run_name_final or metrics is None:
                continue
            overall = metrics.get('전체 마스킹 평가 대상', {})
            if overall.get('cpc') is not None and b_overall.get('cpc'):
                dcpc = overall['cpc'] - b_overall['cpc']
                dcpc_pct = dcpc / b_overall['cpc'] * 100 if b_overall['cpc'] else float('nan')
                drmse = overall.get('rmse', float('nan')) - b_overall.get('rmse', float('nan'))
                drmse_pct = drmse / b_overall['rmse'] * 100 if b_overall.get('rmse') else float('nan')
                a(f"- {run_name}: CPC {dcpc:+.4f} ({dcpc_pct:+.2f}%), RMSE {drmse:+.2f} ({drmse_pct:+.2f}%)")
    else:
        a("- baseline(weighted_mse) 50 epoch 실행이 실패해 비교 불가")
    a("")
    a("## 11. 큰 OD(1,000 이상) 성능 비교")
    a("위 9번 leaderboard의 'OD1000+ RMSE/MAE/CPC' 열 참고.")
    a("")
    a("## 12. 작은 OD(1~100) 성능 비교")
    a("위 9번 leaderboard의 'OD1~100 RMSE/MAE/CPC' 열 참고.")
    a("")
    a("## 13. 동일 동 / 동 간 이동 성능")
    a("위 9번 leaderboard의 '동일동 CPC' / '동간 CPC' 열 참고.")
    a("")
    a("## 14. 실패한 loss 및 실패 원인")
    if failure_rows:
        a("| run_name | stage | loss | attempt | 원인 |")
        a("|---|---|---|---|---|")
        for r in failure_rows:
            a(f"| {r['run_name']} | {r['stage']} | {r['loss']} | {r['attempt']} | {r['error_summary']} |")
    else:
        a("- 실패한 조합 없음")
    a("")
    a("## 15. 추천 loss")
    if ranked_final_new_only:
        rec_name, rec_score = ranked_final_new_only[0]
        rec_label = final_run_meta.get(rec_name, {}).get('loss')
        a(f"- **{rec_label}** (`{rec_name}`)")
    else:
        a("- 추천 없음 — 모든 신규 loss가 안전조건에서 제외되었거나 baseline보다 CPC가 낮음. weighted_mse baseline 유지를 권장.")
    a("")
    a("## 16. 추천 근거")
    a("- 1차 기준(overall masked CPC 최고), 2차 기준(CPC 0.01 이내 동률 시 OD1000+ RMSE 최저), "
      "안전조건(OD1~100 MAE가 baseline 대비 10%+ 악화 시 제외)을 모두 적용한 결과. 상세 수치는 9/10번 참고.")
    a("")
    a("## 17. 큰 OD와 작은 OD를 동시에 개선했는지에 대한 결론")
    a("- 9번 leaderboard의 OD1000+ 열과 OD1~100 열을 함께 보고 판단할 것. "
      "추천 loss가 baseline 대비 OD1000+ RMSE는 개선되었으면서 OD1~100 MAE가 10% 이상 악화되지 않았다면 "
      "\"동시 개선\"으로 볼 수 있음 — 정확한 수치는 위 표를 근거로 판단.")
    a("")
    a("## 18. 단일 seed 실험이라는 한계")
    a("- 모든 정식 실험(Stage 3/4)은 seed=42 단일 시드로만 수행됨. 딥러닝 모델의 seed-to-seed 분산을 고려하지 않았으므로, "
      "여기서의 순위는 하나의 표본에 불과함. 최종 채택 전 최소 2~3개 추가 seed로 재현성을 확인하는 것을 권장.")
    a("")
    a("## 19. 다음 단계에서 추가할 실험")
    a("- 추천 loss에 대한 다중 seed 재현성 검증")
    a("- alpha/lambda 스케줄링(고정값이 아닌 epoch-dependent 스케줄) 자체를 별도 축으로 실험")
    a("- 이번에 제외된(안전조건 위반) 후보들의 하이퍼파라미터를 재조정해 재시도")
    a("- twostage 파이프라인에도 최적 loss를 이식해 비교(이번 실험 범위 밖)")
    a("")

    with open(os.path.join(out_dir, 'final_report.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def get_paths(out_dir):
    return {
        'out_dir': out_dir,
        'runs_dir': os.path.join(out_dir, 'runs'),
        'run_status': os.path.join(out_dir, 'RUN_STATUS.md'),
        'timing_csv': os.path.join(out_dir, 'timing.csv'),
        'failures_csv': os.path.join(out_dir, 'failures.csv'),
        'screening_csv': os.path.join(out_dir, 'screening_leaderboard.csv'),
        'final_csv': os.path.join(out_dir, 'final_leaderboard.csv'),
        'experiment_config': os.path.join(out_dir, 'experiment_config.json'),
        'derived_hparams': os.path.join(out_dir, 'derived_loss_hparams.json'),
        'env_info': os.path.join(out_dir, 'env_info.json'),
        'summary': os.path.join(out_dir, 'summary.json'),
    }


def load_or_create_experiment_config(paths, args, candidate_defs, derived):
    """
    이미 experiment_config.json이 있으면 그대로 재사용한다(Colab에서 세션이 끊겼다 다시
    이어서 --stage 2/3/4를 실행해도 candidates/derived/git_commit 등 설정이 최초 실행과
    동일하게 유지되도록 하기 위함 — 중간에 코드를 바꾸지 않는 한 안전).
    """
    if os.path.exists(paths['experiment_config']):
        with open(paths['experiment_config'], 'r', encoding='utf-8') as f:
            return json.load(f)
    cfg = {
        'git_commit': get_git_commit_hash(),
        'model': args.model,
        'batch_size': args.batch_size,
        'smoke_epochs': args.smoke_epochs,
        'screening_epochs': args.screening_epochs,
        'final_epochs': args.final_epochs,
        'smoke_seed': args.smoke_seed,
        'main_seed': args.main_seed,
        'max_retries': args.max_retries,
        'time_budget_hours': args.time_budget_hours,
        'final_topk': args.final_topk,
        'final_topk_fallback': args.final_topk_fallback,
        'candidates': [{**c, 'params_final': inject_derived_params(c, derived)} for c in candidate_defs],
        'derived_loss_hparams': derived,
        'started_at': time.strftime('%Y-%m-%d %H:%M:%S %z'),
    }
    with open(paths['experiment_config'], 'w', encoding='utf-8') as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    return cfg


def stage1_ok(paths, cfg):
    run_dir = os.path.join(paths['runs_dir'], 'stage1_weighted_mse_baseline_smoke')
    ckpt, csv_path, _ = find_result_files(run_dir, cfg['model'], 'weighted_mse', cfg['smoke_seed'])
    return os.path.exists(ckpt) and os.path.exists(csv_path)


def get_passed_stage2_candidates(paths, cfg, candidate_defs):
    """디스크에 남은 stage2 결과를 스캔해서 통과 후보를 재구성(프로세스 재시작에도 안전)."""
    passed = []
    for c in candidate_defs:
        run_dir = os.path.join(paths['runs_dir'], f"stage2_{c['label']}_smoke")
        ckpt, csv_path, _ = find_result_files(run_dir, cfg['model'], c['loss'], cfg['smoke_seed'])
        if os.path.exists(ckpt) and os.path.exists(csv_path):
            passed.append(c)
    return passed


def do_stage1(paths, cfg, python_exe):
    append_run_status(paths['run_status'], [log("Stage 1 시작: weighted_mse baseline smoke test")])
    baseline_smoke_name = 'stage1_weighted_mse_baseline_smoke'
    ok, run_dir, csv_path = run_one(
        python_exe, paths['runs_dir'], 'stage1', baseline_smoke_name, cfg['model'],
        cfg['smoke_epochs'], cfg['batch_size'], 'weighted_mse', {}, cfg['smoke_seed'],
        cfg['max_retries'], paths['timing_csv'], paths['failures_csv'])
    if not ok:
        append_run_status(paths['run_status'], [
            log("Stage 1 실패: weighted_mse baseline smoke test가 실패했습니다."),
            log(f"로그 확인: {os.path.join(paths['runs_dir'], baseline_smoke_name, 'train.log')}"),
        ])
        return False
    append_run_status(paths['run_status'], [log("Stage 1 완료: weighted_mse baseline smoke test 성공")])
    return True


def do_stage2(paths, cfg, candidate_defs, derived, python_exe):
    if not stage1_ok(paths, cfg):
        append_run_status(paths['run_status'], [
            log("Stage 2 중단: Stage 1이 아직 성공 상태가 아닙니다. 먼저 --stage 1을 실행하세요.")])
        return []
    append_run_status(paths['run_status'], [log(f"Stage 2 시작: 신규 loss {len(candidate_defs)}개 조합 smoke test")])
    passed = []
    for c in candidate_defs:
        params = inject_derived_params(c, derived)
        run_name = f"stage2_{c['label']}_smoke"
        ok, _, _ = run_one(
            python_exe, paths['runs_dir'], 'stage2', run_name, cfg['model'],
            cfg['smoke_epochs'], cfg['batch_size'], c['loss'], params, cfg['smoke_seed'],
            cfg['max_retries'], paths['timing_csv'], paths['failures_csv'])
        if ok:
            passed.append(c)
        append_run_status(paths['run_status'], [log(f"Stage 2 [{c['label']}] {'통과' if ok else '실패(제외)'}")])
    append_run_status(paths['run_status'], [
        log(f"Stage 2 완료: {len(passed)}/{len(candidate_defs)}개 통과 "
            f"({', '.join(c['label'] for c in passed) if passed else '없음'})")])
    return passed


def do_stage3(paths, cfg, candidate_defs, derived, python_exe):
    passed_candidates = get_passed_stage2_candidates(paths, cfg, candidate_defs)
    if not passed_candidates:
        append_run_status(paths['run_status'], [
            log("Stage 3 경고: Stage 2를 통과한 신규 후보가 없습니다. baseline만 screening합니다.")])
    append_run_status(paths['run_status'], [log("Stage 3 시작: 10 epoch screening (baseline + 통과 후보)")])
    screening_entries = [{'loss': 'weighted_mse', 'label': 'weighted_mse_baseline', 'params': {}}] + passed_candidates
    screening_metrics = {}
    screening_run_meta = {}
    for c in screening_entries:
        params = inject_derived_params(c, derived) if c['label'] != 'weighted_mse_baseline' else {}
        run_name = f"stage3_{c['label']}_10ep"
        ok, run_dir, csv_path = run_one(
            python_exe, paths['runs_dir'], 'stage3', run_name, cfg['model'],
            cfg['screening_epochs'], cfg['batch_size'], c['loss'], params, cfg['main_seed'],
            cfg['max_retries'], paths['timing_csv'], paths['failures_csv'])
        screening_run_meta[run_name] = {'loss': c['loss'], 'label': c['label'],
                                         'params': json.dumps(params, ensure_ascii=False)}
        screening_metrics[run_name] = load_official_metrics(csv_path) if ok else None
        append_run_status(paths['run_status'], [log(f"Stage 3 [{c['label']}] {'완료' if ok else '실패'}")])

    write_leaderboard_csv(paths['screening_csv'], screening_metrics, screening_run_meta)
    baseline_run_name = 'stage3_weighted_mse_baseline_10ep'
    ranked_full, excluded_full = rank_candidates(dict(screening_metrics), baseline_run_name=baseline_run_name)
    ranked_new_only = [(name, s) for name, s in ranked_full if name != baseline_run_name]
    top_new = ranked_new_only[:cfg['final_topk']]
    top_labels = [screening_run_meta[name]['label'] for name, _ in top_new]
    append_run_status(paths['run_status'], [
        log(f"Stage 3 완료: 10 epoch screening 상위 후보 = {top_labels}"),
        log(f"Stage 3 안전조건 제외: {excluded_full}"),
    ])
    return screening_metrics, screening_run_meta


def do_stage4(paths, cfg, candidate_defs, derived, python_exe):
    if not os.path.exists(paths['screening_csv']):
        append_run_status(paths['run_status'], [
            log("Stage 4 중단: screening_leaderboard.csv가 없습니다. 먼저 --stage 3을 실행하세요.")])
        return

    screening_metrics, screening_run_meta = read_leaderboard_csv(paths['screening_csv'])
    baseline_run_name = 'stage3_weighted_mse_baseline_10ep'
    ranked_full, excluded_full = rank_candidates(dict(screening_metrics), baseline_run_name=baseline_run_name)
    ranked_new_only = [(name, s) for name, s in ranked_full if name != baseline_run_name]
    top_new = ranked_new_only[:cfg['final_topk']]

    timing_rows = read_csv_rows(paths['timing_csv'])
    per_epoch_times = [float(r['epoch_time_sec_avg']) for r in timing_rows
                        if r.get('stage') == 'stage3' and r.get('success') == 'True' and r.get('epoch_time_sec_avg')]
    avg_epoch_time = sum(per_epoch_times) / len(per_epoch_times) if per_epoch_times else None

    final_topk = cfg['final_topk']
    if avg_epoch_time is not None:
        n_configs = 1 + min(len(top_new), cfg['final_topk'])
        projected_hours = n_configs * cfg['final_epochs'] * avg_epoch_time / 3600.0
        append_run_status(paths['run_status'], [
            log(f"50 epoch 예상 총 시간: 약 {projected_hours:.2f}시간 "
                f"(baseline+상위 {min(len(top_new), cfg['final_topk'])}개, epoch당 평균 {avg_epoch_time:.1f}s 기준)")])
        if projected_hours > cfg['time_budget_hours']:
            final_topk = cfg['final_topk_fallback']
            append_run_status(paths['run_status'], [
                log(f"시간 예산({cfg['time_budget_hours']}h) 초과 예상 → 최종 후보를 상위 {final_topk}개로 축소")])

    final_new_candidates = top_new[:final_topk]
    append_run_status(paths['run_status'], [
        log(f"Stage 4 시작: weighted_mse baseline + 상위 {len(final_new_candidates)}개, 50 epoch")])

    final_entries = [{'loss': 'weighted_mse', 'label': 'weighted_mse_baseline', 'params': {}}]
    for name, _ in final_new_candidates:
        meta = screening_run_meta[name]
        c = next(c for c in candidate_defs if c['label'] == meta['label'])
        final_entries.append(c)

    final_metrics = {}
    final_run_meta = {}
    for c in final_entries:
        params = inject_derived_params(c, derived) if c['label'] != 'weighted_mse_baseline' else {}
        run_name = f"stage4_{c['label']}_50ep"
        ok, run_dir, csv_path = run_one(
            python_exe, paths['runs_dir'], 'stage4', run_name, cfg['model'],
            cfg['final_epochs'], cfg['batch_size'], c['loss'], params, cfg['main_seed'],
            cfg['max_retries'], paths['timing_csv'], paths['failures_csv'])
        final_run_meta[run_name] = {'loss': c['loss'], 'label': c['label'],
                                     'params': json.dumps(params, ensure_ascii=False)}
        final_metrics[run_name] = load_official_metrics(csv_path) if ok else None
        append_run_status(paths['run_status'], [
            log(f"Stage 4 [{c['label']}] {'완료' if ok else '실패'} — 다음 실험 계속 진행")])

    write_leaderboard_csv(paths['final_csv'], final_metrics, final_run_meta)

    baseline_run_name_final = 'stage4_weighted_mse_baseline_50ep'
    ranked_final, excluded_final = rank_candidates(final_metrics, baseline_run_name=baseline_run_name_final)
    ranked_final_new_only = [(n, s) for n, s in ranked_final if n != baseline_run_name_final]
    recommendation = ranked_final_new_only[0] if ranked_final_new_only else None

    summary = {
        'final_metrics': final_metrics, 'final_run_meta': final_run_meta,
        'screening_metrics': screening_metrics, 'screening_run_meta': screening_run_meta,
        'excluded_stage3': excluded_full, 'excluded_stage4': excluded_final,
        'recommendation_run': recommendation[0] if recommendation else None,
        'baseline_run_name_final': baseline_run_name_final,
    }
    with open(paths['summary'], 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    append_run_status(paths['run_status'], [
        log("Stage 4 완료: 최종 leaderboard 작성 완료"),
        log(f"안전조건 제외(최종): {excluded_final}"),
        log(f"1순위 추천: {recommendation[0] if recommendation else '없음(모든 신규 loss가 baseline보다 못하거나 제외됨)'}"),
    ])


def do_report(paths, cfg, candidate_defs, python_exe):
    if not os.path.exists(paths['final_csv']):
        append_run_status(paths['run_status'], [
            log("리포트 작성 중단: final_leaderboard.csv가 없습니다. 먼저 --stage 4를 실행하세요.")])
        return

    final_metrics, final_run_meta = read_leaderboard_csv(paths['final_csv'])
    if os.path.exists(paths['screening_csv']):
        screening_metrics, screening_run_meta = read_leaderboard_csv(paths['screening_csv'])
    else:
        screening_metrics, screening_run_meta = {}, {}

    summary = {}
    if os.path.exists(paths['summary']):
        with open(paths['summary'], 'r', encoding='utf-8') as f:
            summary = json.load(f)
    excluded_stage3 = summary.get('excluded_stage3', {})
    excluded_stage4 = summary.get('excluded_stage4', {})
    baseline_run_name_final = summary.get('baseline_run_name_final', 'stage4_weighted_mse_baseline_50ep')

    passed_stage2 = [c['label'] for c in get_passed_stage2_candidates(paths, cfg, candidate_defs)]

    if os.path.exists(paths['env_info']):
        with open(paths['env_info'], 'r', encoding='utf-8') as f:
            env_info = json.load(f)
    else:
        env_info = capture_env_info(python_exe)
        with open(paths['env_info'], 'w', encoding='utf-8') as f:
            json.dump(env_info, f, ensure_ascii=False, indent=2)

    timing_rows = read_csv_rows(paths['timing_csv'])
    failure_rows = read_csv_rows(paths['failures_csv'])
    total_elapsed_hours = sum(
        float(r['wall_time_sec']) for r in timing_rows if r.get('wall_time_sec')
    ) / 3600.0

    ranked_final, _ = rank_candidates(final_metrics, baseline_run_name=baseline_run_name_final)
    ranked_final_new_only = [(n, s) for n, s in ranked_final if n != baseline_run_name_final]

    try:
        write_final_report(
            paths['out_dir'], cfg, env_info, candidate_defs,
            passed_stage2, screening_metrics, screening_run_meta,
            final_metrics, final_run_meta, excluded_stage3, excluded_stage4,
            ranked_final_new_only, baseline_run_name_final,
            timing_rows, failure_rows, total_elapsed_hours)
        append_run_status(paths['run_status'], [log("final_report.md 작성 완료")])
    except Exception as e:
        append_run_status(paths['run_status'], [
            log(f"final_report.md 작성 중 오류: {e}\n{traceback.format_exc()}")])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--python', type=str, required=True, help='학습에 사용할 python 실행 파일 경로')
    parser.add_argument('--model', type=str, default='mae1')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--smoke-epochs', type=int, default=2)
    parser.add_argument('--screening-epochs', type=int, default=10)
    parser.add_argument('--final-epochs', type=int, default=50)
    parser.add_argument('--smoke-seed', type=int, default=999)
    parser.add_argument('--main-seed', type=int, default=42)
    parser.add_argument('--max-retries', type=int, default=2)
    parser.add_argument('--time-budget-hours', type=float, default=8.0)
    parser.add_argument('--final-topk', type=int, default=3)
    parser.add_argument('--final-topk-fallback', type=int, default=2)
    parser.add_argument('--only-candidates', type=str, default=None,
                         help='콤마로 구분된 candidate label만 실행(디버그/quick-test용)')
    parser.add_argument('--stage', type=str, default='all', choices=['1', '2', '3', '4', 'report', 'all'],
                         help="Colab처럼 세션이 끊길 수 있는 환경에서 stage별로 셀을 나눠 실행/재개하기 위한 옵션. "
                              "각 stage는 실행 결과를 디스크에서 다시 읽어 이전 stage의 산출물을 재구성하므로, "
                              "중간에 프로세스가 죽어도 같은 --out-dir로 다시 실행하면 이미 끝난 run은 건너뛰고 "
                              "이어서 진행한다. 기본값 all은 1→2→3→4→report를 한 프로세스에서 순서대로 실행(로컬용).")
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    paths = get_paths(out_dir)
    os.makedirs(paths['runs_dir'], exist_ok=True)

    if not os.path.exists(paths['derived_hparams']):
        append_run_status(paths['run_status'], [
            log("치명적 오류: derived_loss_hparams.json이 없어 실행 중단 "
                "(먼저 scripts/analyze_target_distribution.py --out-dir 를 실행하세요)")])
        sys.exit(1)
    with open(paths['derived_hparams'], 'r', encoding='utf-8') as f:
        derived = json.load(f)

    candidate_defs = build_candidate_defs()
    if args.only_candidates:
        only = set(args.only_candidates.split(','))
        candidate_defs = [c for c in candidate_defs if c['label'] in only]

    cfg = load_or_create_experiment_config(paths, args, candidate_defs, derived)

    if not os.path.exists(paths['env_info']):
        env_info = capture_env_info(args.python)
        with open(paths['env_info'], 'w', encoding='utf-8') as f:
            json.dump(env_info, f, ensure_ascii=False, indent=2)
        append_run_status(paths['run_status'], [log(f"환경 정보: {env_info}")])

    t_start = time.time()

    if args.stage in ('1', 'all'):
        if not do_stage1(paths, cfg, args.python):
            sys.exit(1)

    if args.stage in ('2', 'all'):
        if not stage1_ok(paths, cfg):
            append_run_status(paths['run_status'], [log("중단: Stage 1이 성공 상태가 아닙니다. 먼저 --stage 1을 실행하세요.")])
            sys.exit(1)
        do_stage2(paths, cfg, candidate_defs, derived, args.python)

    if args.stage in ('3', 'all'):
        do_stage3(paths, cfg, candidate_defs, derived, args.python)

    if args.stage in ('4', 'all'):
        do_stage4(paths, cfg, candidate_defs, derived, args.python)

    if args.stage in ('report', 'all'):
        do_report(paths, cfg, candidate_defs, args.python)

    elapsed_h = (time.time() - t_start) / 3600.0
    append_run_status(paths['run_status'], [log(f"[--stage {args.stage}] 이번 호출 소요 시간: {elapsed_h:.2f}시간")])


if __name__ == '__main__':
    main()
