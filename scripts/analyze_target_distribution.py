"""
SpatialODMAE1 학습 target의 분포를 분석한다. 두 가지 분포를 함께 계산한다.

1) "flatten" 분포: train_indices x train_indices(legacy) 또는 strict_train_indices x
   strict_train_indices(strict) 부분행렬을 단순히 펼친 값의 분포. 서술적 통계(0 비율,
   percentile, 동일동/동간 비교 등)에 사용한다.
2) "sampled" 분포: `dataset.py`의 실제 masking curriculum(ODDataset.__getitem__)을 고정
   seed로 여러 번 호출해, 그 결과로 실제 loss 계산에 들어가는 target 값들만 모아 집계한
   분포. BinBalancedMSELoss의 bin_freq는 이 sampled 분포를 사용해야 한다 — 단순 flatten은
   "학습이 실제로 보는" 분포와 다르다(masking이 거리 기반으로 편향돼 있기 때문).

- dataset.py를 그대로 import해서 쓴다(재구현하지 않음).
- real scale(X_OD, 통행량 원 단위) 기준으로 분석한다.
- --protocol strict를 주면 dataset.py의 protocol='strict' 경로를 그대로 사용해
  test_indices/strict_val_indices를 모두 배제한 분포를 계산한다.

출력:
  <out_dir>/target_distribution.csv
  <out_dir>/target_distribution.md
  <out_dir>/derived_loss_hparams.json   (신규 loss들의 데이터 기반 하이퍼파라미터 기본값)
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'model'))

from dataset import ODDataset  # noqa: E402

BIN_NAMES = ['실제값 0', '실제값 0 초과 100 이하', '실제값 100 초과 1000 미만', '실제값 1000 이상']


def bin_index(values):
    # 0 / (0,100] / (100,1000) / [1000,∞) — 겹침·빈틈 없는 완전 분할.
    # OD 값이 소수로도 존재하므로 정수 경계(>=1, >=101 등)를 쓰면 (0,1)/(100,101)/(999,1000)
    # 구간의 값이 어느 bin에도 속하지 못하는 문제가 있어 이렇게 정의함
    # (src/eval_utils.py, src/model/loss.py의 BinBalancedMSELoss와 동일한 경계 정의로 통일).
    idx = np.zeros_like(values, dtype=np.int64)
    idx[(values > 0) & (values <= 100)] = 1
    idx[(values > 100) & (values < 1000)] = 2
    idx[values >= 1000] = 3
    return idx


def compute_sampled_distribution(ds, protocol, n_samples, seed):
    """
    dataset.py의 실제 masking curriculum(__getitem__)을 고정 seed로 n_samples번 호출해
    실제 loss 대상이 되는 target(real scale) 값들을 모아 bin 분포를 계산한다.
    """
    np.random.seed(seed)
    N = ds.num_nodes
    pooled = []
    for _ in range(n_samples):
        item = ds[0]  # idx는 사용되지 않고 매번 새로 랜덤 샘플링됨(dataset.py 참고)
        mask = item['mask'].numpy()
        y_od_log = item['y_OD'].numpy()  # isLogScale=True로 로드했다면 log1p 스케일
        mask_2d = mask[:, None] | mask[None, :]
        if protocol == 'strict':
            mask_2d = mask_2d & ds.strict_train_safe_od_mask
        target_log = y_od_log[mask_2d]
        target_real = np.expm1(target_log)
        pooled.append(target_real)
    pooled = np.concatenate(pooled) if pooled else np.array([])
    return pooled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--protocol', type=str, default='legacy', choices=['legacy', 'strict'])
    parser.add_argument('--alpha', type=float, default=1.5,
                         help='weighted_mse_fixed의 alpha(참고 후보, 기여 비중 근사 계산용)')
    parser.add_argument('--n-mask-samples', type=int, default=300,
                         help='masking sampler 분포를 추정하기 위해 __getitem__을 호출할 횟수')
    parser.add_argument('--sample-seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading dataset (protocol={args.protocol}, real scale for flatten stats)...")
    ds_real = ODDataset(mode='train', channel=1, isLogScale=False, protocol=args.protocol)

    if args.protocol == 'strict':
        pool_idx = ds_real.strict_train_indices
    else:
        pool_idx = ds_real.train_indices

    sub = ds_real.X_OD[np.ix_(pool_idx, pool_idx)].astype(np.float64)
    N = sub.shape[0]
    values = sub.flatten()
    total_count = values.size

    zero_mask = values == 0
    pos_values = values[~zero_mask]

    zero_ratio = float(zero_mask.mean())
    pos_mean = float(pos_values.mean()) if pos_values.size else float('nan')
    pos_median = float(np.median(pos_values)) if pos_values.size else float('nan')
    pos_std = float(pos_values.std()) if pos_values.size else float('nan')

    percentiles = [90, 95, 99, 99.5, 99.9]
    pct_all = {p: float(np.percentile(values, p)) for p in percentiles}
    pct_pos = {p: float(np.percentile(pos_values, p)) for p in percentiles} if pos_values.size else {p: float('nan') for p in percentiles}

    bins = bin_index(values)
    bin_counts = [int((bins == i).sum()) for i in range(4)]
    bin_ratios_flatten = [c / total_count for c in bin_counts]

    diag_values = np.diag(sub)
    off_diag_mask = ~np.eye(N, dtype=bool)
    offdiag_values = sub[off_diag_mask]
    same_dong_count = int(diag_values.size)
    same_dong_mean = float(diag_values.mean())
    diff_dong_count = int(offdiag_values.size)
    diff_dong_mean = float(offdiag_values.mean())

    log_values = np.log1p(values)
    weights = 1.0 + args.alpha * log_values
    total_weight = weights.sum()
    bin_weight_share = [float(weights[bins == i].sum() / total_weight) for i in range(4)]

    # ---- sampled 분포: 실제 masking curriculum을 재사용해서 계산 ----
    print(f"Sampling actual masking curriculum distribution "
          f"(protocol={args.protocol}, n={args.n_mask_samples}, seed={args.sample_seed})...")
    ds_log = ODDataset(mode='train', channel=1, isLogScale=True, protocol=args.protocol)
    sampled_values = compute_sampled_distribution(ds_log, args.protocol, args.n_mask_samples, args.sample_seed)
    sampled_total = sampled_values.size
    sampled_bins = bin_index(sampled_values)
    sampled_bin_counts = [int((sampled_bins == i).sum()) for i in range(4)]
    sampled_bin_ratios = [c / sampled_total if sampled_total else 0.0 for c in sampled_bin_counts]
    sampled_zero_ratio = sampled_bin_ratios[0]

    # ---- CSV ----
    rows = [
        {'metric': 'protocol', 'value': args.protocol},
        {'metric': 'total_od_count_flatten', 'value': total_count},
        {'metric': 'zero_ratio_flatten', 'value': zero_ratio},
        {'metric': 'positive_mean_flatten', 'value': pos_mean},
        {'metric': 'positive_median_flatten', 'value': pos_median},
        {'metric': 'positive_std_flatten', 'value': pos_std},
        {'metric': 'total_od_count_sampled', 'value': sampled_total},
        {'metric': 'zero_ratio_sampled', 'value': sampled_zero_ratio},
        {'metric': 'n_mask_samples', 'value': args.n_mask_samples},
        {'metric': 'sample_seed', 'value': args.sample_seed},
    ]
    for p in percentiles:
        rows.append({'metric': f'percentile_{p}_all_flatten', 'value': pct_all[p]})
    for p in percentiles:
        rows.append({'metric': f'percentile_{p}_positive_only_flatten', 'value': pct_pos[p]})
    for name, cnt, ratio, wshare in zip(BIN_NAMES, bin_counts, bin_ratios_flatten, bin_weight_share):
        rows.append({'metric': f'flatten_count[{name}]', 'value': cnt})
        rows.append({'metric': f'flatten_ratio[{name}]', 'value': ratio})
        rows.append({'metric': f'weighted_mse_fixed_loss_share[{name}]', 'value': wshare})
    for name, cnt, ratio in zip(BIN_NAMES, sampled_bin_counts, sampled_bin_ratios):
        rows.append({'metric': f'sampled_count[{name}]', 'value': cnt})
        rows.append({'metric': f'sampled_ratio[{name}]', 'value': ratio})
    rows.append({'metric': 'same_dong_count_flatten', 'value': same_dong_count})
    rows.append({'metric': 'same_dong_mean_flatten', 'value': same_dong_mean})
    rows.append({'metric': 'diff_dong_count_flatten', 'value': diff_dong_count})
    rows.append({'metric': 'diff_dong_mean_flatten', 'value': diff_dong_mean})

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.out_dir, f'target_distribution_{args.protocol}.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"Saved {csv_path}")

    # ---- MD ----
    md_lines = []
    a = md_lines.append
    a("# Target 분포 분석 (SpatialODMAE1 학습 대상)")
    a("")
    a(f"- protocol: **{args.protocol}**")
    a(f"- flatten 분석 대상: {'strict_train_indices' if args.protocol=='strict' else 'train_indices'} x 동일 "
      f"부분행렬 (N={N}, 총 {total_count:,}개 OD 쌍)")
    a(f"- sampled 분석 대상: `dataset.py`의 실제 masking curriculum을 seed={args.sample_seed}로 "
      f"{args.n_mask_samples}회 호출해 모은 실제 loss 대상 값 (총 {sampled_total:,}개)")
    a("")
    a("## 기본 통계 (flatten 기준)")
    a(f"- 0의 비율: {zero_ratio:.4%}")
    a(f"- 양수 OD 평균: {pos_mean:,.2f}")
    a(f"- 양수 OD 중앙값: {pos_median:,.2f}")
    a(f"- 양수 OD 표준편차: {pos_std:,.2f}")
    a("")
    a("## 분위수 (flatten 기준)")
    a("| 분위수 | 전체(0 포함) | 양수만 |")
    a("|---|---|---|")
    for p in percentiles:
        a(f"| {p}% | {pct_all[p]:,.2f} | {pct_pos[p]:,.2f} |")
    a("")
    a("## 구간별 분포 비교: flatten vs 실제 masking sampler(sampled)")
    a("BinBalancedMSELoss의 bin_freq는 **sampled** 열을 사용한다(단순 flatten이 아님 — "
      "masking curriculum이 거리 기반이라 flatten과 다를 수 있음).")
    a("| 구간 | flatten 비율 | sampled 비율 | weighted_mse_fixed(alpha={:.1f}) 기여 비중* |".format(args.alpha))
    a("|---|---|---|---|")
    for name, r_flat, r_samp, wshare in zip(BIN_NAMES, bin_ratios_flatten, sampled_bin_ratios, bin_weight_share):
        a(f"| {name} | {r_flat:.4%} | {r_samp:.4%} | {wshare:.4%} |")
    a("")
    a("\\* weighted_mse_fixed 기여 비중은 flatten 분포 기준 근사치(참고용, historical baseline이 아님).")
    a("")
    a("## 동일 행정동 내부 vs 동 간 이동 (flatten 기준)")
    a("| 구분 | 표본 수 | 평균 |")
    a("|---|---|---|")
    a(f"| 동일 행정동 내부(대각) | {same_dong_count:,} | {same_dong_mean:,.2f} |")
    a(f"| 서로 다른 행정동 간(비대각) | {diff_dong_count:,} | {diff_dong_mean:,.2f} |")
    a("")

    md_path = os.path.join(args.out_dir, f'target_distribution_{args.protocol}.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    print(f"Saved {md_path}")

    # ---- 신규 loss 하이퍼파라미터 기본값(데이터 기반) ----
    # DualScaleMSELoss.global_scale: 배치별 폴백과 동일한 정의(RMS, 0 포함 전체)를 전역(flatten) 데이터로 계산.
    global_scale_rms = float(np.sqrt(np.mean(values.astype(np.float64) ** 2)))
    # BinBalancedMSELoss.bin_freq: sampled 분포 사용(요청사항).
    # positive_relative_hybrid.tau: 1차 실행에서는 tau=50 고정(loss.py 참고, 데이터 median이
    # 너무 작아 median 자체를 tau로 쓰지 않기로 함) — 여기서는 참고용으로 median만 기록한다.
    derived = {
        'dual_scale_mse': {'global_scale': global_scale_rms},
        'bin_balanced_mse': {'bin_freq': sampled_bin_ratios},
        'positive_relative_hybrid': {'tau': 50.0},
        'source': {
            'protocol': args.protocol,
            'zero_ratio_flatten': zero_ratio,
            'zero_ratio_sampled': sampled_zero_ratio,
            'positive_median_flatten': pos_median,
            'global_scale_rms_flatten': global_scale_rms,
            'bin_ratios_flatten': dict(zip(BIN_NAMES, bin_ratios_flatten)),
            'bin_ratios_sampled': dict(zip(BIN_NAMES, sampled_bin_ratios)),
            'n_mask_samples': args.n_mask_samples,
            'sample_seed': args.sample_seed,
        },
    }
    hparam_path = os.path.join(args.out_dir, f'derived_loss_hparams_{args.protocol}.json')
    with open(hparam_path, 'w', encoding='utf-8') as f:
        json.dump(derived, f, ensure_ascii=False, indent=2)
    print(f"Saved {hparam_path}")


if __name__ == '__main__':
    main()
