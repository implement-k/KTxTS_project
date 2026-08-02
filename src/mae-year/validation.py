import os
import sys
import torch, time
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from evaluation.fixed_eval_utils import apply_merge_events

def format_minutes(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s"

def _eval_one_sample(args):
    """(model, base_data, ...) 받아 단일 샘플 평가 -> dict 반환"""
    model, base_data, year_label, city_name, task, split_name, mask_indices, merge_events, device = args
    
    try:
        # validation에서는 test 동 정보를 입력에서 숨겨야 치팅이 아니다.
        # test에서는 test 동 자체가 평가 대상이므로 전체 feature를 hide하면 안 된다.
        holdout_indices = base_data['test_indices'] if split_name == 'val' else []
        sample = apply_merge_events(
            base_data,
            mask_indices,
            merge_events,
            hide_indices=holdout_indices,
            imputation_values= None
        )
        
        # MAE 모델 추론 준비
        x_static = sample['X_static'].float().unsqueeze(0).to(device) # norm
        x_dist = sample['X_dist'].float().unsqueeze(0).to(device) # log1p
        mask_t = sample['mask'].unsqueeze(0).to(device) 
        x_od_masked = sample['X_OD_masked'].float().unsqueeze(0).to(device) # log1p
        a_spatial = sample['A_spatial'].float().unsqueeze(0).to(device) 
        active_node_mask = sample['active_node_mask'].unsqueeze(0).to(device) # 병합된 노드 제외

        # 병렬 스레드 환경이므로 안전하게
        with torch.no_grad():
            pred = model(x_static, x_od_masked, x_dist, a_spatial, mask_t, active_node_mask)
        
        if not torch.isfinite(pred).all():
            raise FloatingPointError("model prediction contains NaN or Inf")

        # log 변환을 원복. Random/untrained outputs can otherwise overflow.
        T_pred = torch.expm1(pred[0].clamp(max=20.0)).cpu().numpy()
        y_od = sample['y_OD_raw'].cpu().numpy()
        if not np.isfinite(y_od).all():
            raise FloatingPointError("validation target contains NaN or Inf")

        eval_indices = np.array(mask_indices)
        N = y_od.shape[0]
        
        eval_mask_2d = np.zeros((N, N), dtype=bool)
        eval_mask_2d[:, eval_indices] = True
        eval_mask_2d[eval_indices, :] = True
        
        # active_node_mask를 반영하여 병합된 노드(hide)는 제외
        active_m2d = active_node_mask.cpu().numpy().reshape(-1, 1) & active_node_mask.cpu().numpy().reshape(1, -1)
        valid_cells = eval_mask_2d & active_m2d
        
        diagonal = np.eye(N, dtype=bool)

        def region_metrics(region):
            truth = y_od[region].astype(np.float64, copy=False)
            prediction = np.maximum(T_pred[region], 0).astype(np.float64, copy=False)
            if truth.size == 0:
                return {'rmse': 0.0, 'cpc': 0.0, 'prmse': 0.0, 'cell_count': 0}
            difference = truth - prediction
            rmse = float(np.sqrt(np.mean(np.square(difference))))
            denominator = float(np.sum(truth) + np.sum(prediction))
            cpc = float(2.0 * np.sum(np.minimum(truth, prediction)) / denominator) if denominator > 0 else 0.0
            mean_target = float(np.mean(truth))
            return {
                'rmse': rmse,
                'cpc': cpc,
                'prmse': rmse / mean_target if mean_target > 0 else 0.0,
                'cell_count': int(truth.size),
            }

        regions = {
            'overall': valid_cells,
            'offdiag': valid_cells & ~diagonal,
            'diagonal': valid_cells & diagonal,
        }
        metrics = {name: region_metrics(region) for name, region in regions.items()}
        record = {'year': year_label, 'city': city_name, 'task': task, 'split': split_name}
        for region, values in metrics.items():
            prefix = '' if region == 'overall' else f'{region}_'
            for key in ('rmse', 'cpc', 'prmse'):
                record[f'{prefix}{key}'] = values[key]
            record[f'{region}_cell_count'] = values['cell_count']
        return record
    except Exception as e:
        raise RuntimeError(
            f"validation sample failed ({year_label}/{city_name}/task={task})"
        ) from e


def validate_metadata_completeness(val_meta):
    """Require every city to contain non-empty Task 0-4 sample lists."""
    if not val_meta:
        raise ValueError("fixed validation metadata is empty")
    for city_name, tasks in val_meta.items():
        missing = [task for task in range(5) if task not in tasks or not tasks[task]]
        if missing:
            raise ValueError(f"fixed validation is incomplete for {city_name}: tasks {missing}")


def evaluate_and_report(base_data, val_meta, model, year_label, split_name,  n_workers=4, device=None,
                        max_samples_per_group=None):
    """ThreadPoolExecutor로 샘플병 병렬 평가"""
    
    validate_metadata_completeness(val_meta)
    job_args = []
    for task in [0, 1, 2, 3, 4]:
        for city_name, val_meta_task_list in val_meta.items():
            samples = val_meta_task_list[task]
            if max_samples_per_group is not None:
                samples = samples[:max_samples_per_group]
            if not samples:
                raise ValueError(
                    f"fixed validation sample limit leaves no samples for {city_name}/task={task}"
                )
            for meta in samples:
                job_args.append((
                    model, base_data, year_label, city_name, task, split_name,
                    meta['mask_indices'], meta['merge_events'], device
                ))
    
    total = len(job_args)
    print(f"  [{year_label}/{split_name}] {total}개 샘플 평가 (threads={n_workers})...")
    
    results: list = [None] * total
    start_time = time.time()
    progress_every = 30
    
    # model.eval() 상태인지 확인 필요
    was_training = model.training
    model.eval()
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_eval_one_sample, args): i
            for i, args in enumerate(job_args)
        }
        for done_count, future in enumerate(as_completed(futures), start=1):
            results[futures[future]] = future.result()

            if done_count % progress_every == 0 or done_count == total:
                elapsed = time.time() - start_time
                rate = done_count / elapsed if elapsed > 0 else 0.0
                remaining = (total - done_count) / rate if rate > 0 else 0.0
                pct = done_count / total * 100 if total > 0 else 100.0
                done_records = [r for r in results if r is not None]
                if done_records:
                    cpc_mean = np.mean([r['cpc'] for r in done_records])
                    rmse_mean = np.mean([r['rmse'] for r in done_records])
                    prmse_mean = np.mean([r['prmse'] for r in done_records])
                    metric_text = (
                        f" | 누적 CPC {cpc_mean:.4f} "
                        f"| RMSE {rmse_mean:.4f} "
                        f"| %RMSE {prmse_mean:.4f}"
                    )
                else:
                    metric_text = ""
                print(
                    f"  [{year_label}/{split_name}] 진행 {done_count}/{total} "
                    f"({pct:.1f}%){metric_text} "
                    f"| 경과 {format_minutes(elapsed)} | 예상 남음 {format_minutes(remaining)}"
                )
    
    if was_training:
        model.train()
        
    records = [r for r in results if r is not None]
    if len(records) != total:
        raise RuntimeError(f"validation completeness failure: {len(records)}/{total} samples")
    print(f"  [{year_label}/{split_name}] 완료: {len(records)}/{total}")
    return records

def summarize_results(records, group_keys, label):
    """group_keys에 따라 records를 그룹화하고, 각 그룹의 평균과 표준편차를 계산하여 요약"""
    grouped = defaultdict(lambda: defaultdict(list))
    
    for record in records:
        key = tuple(record[key] for key in group_keys)
        for metric in ('rmse', 'cpc', 'prmse'):
            grouped[key][metric].append(record[metric])
            
    print(f"\n=== {label} ===")
    for key in sorted(grouped.keys()):
        key_str = ", ".join(f"{k}={v}" for k, v in zip(group_keys, key))
        n = len(grouped[key]['cpc'])
        cpc_mean, cpc_std = np.mean(grouped[key]['cpc']), np.std(grouped[key]['cpc'])
        rmse_mean, rmse_std = np.mean(grouped[key]['rmse']), np.std(grouped[key]['rmse'])
        prmse_mean, prmse_std = np.mean(grouped[key]['prmse']), np.std(grouped[key]['prmse'])
        print(f"[{key_str}] (n={n}) "
              f"CPC={cpc_mean:.4f}±{cpc_std:.4f}  "
              f"RMSE={rmse_mean:.4f}±{rmse_std:.4f}  "
              f"%RMSE={prmse_mean:.4f}±{prmse_std:.4f}")
