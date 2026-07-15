"""
SpatialODMAE1 학습 target(train_indices x train_indices OD 부분행렬)의 분포를 분석한다.

- train.py/dataset.py를 그대로 사용해 실제 학습에 쓰이는 것과 동일한 데이터를 로드한다
  (config.py, dataset.py를 전혀 수정하지 않고 그대로 import).
- 분석 대상은 train_indices x train_indices 부분행렬이다: 학습 시 masking은 항상
  train_indices 안에서만 이뤄지고(dataset.py의 __getitem__), y_OD도 test 지역을 포함한
  전체 행렬이지만 실제 loss가 걸리는 마스킹 대상은 train 지역 동끼리의 조합이므로,
  "학습 target 분포"를 가장 잘 대표하는 부분집합으로 이 서브매트릭스를 선택했다.
- real scale(X_OD, 통행량 원 단위) 기준으로 분석한다.

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--alpha', type=float, default=1.5,
                         help='기존 Weighted MSE의 alpha(고정 baseline과 동일한 값 사용)')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading dataset (real scale, isLogScale=False)...")
    ds = ODDataset(mode='train', channel=1, isLogScale=False)

    train_idx = ds.train_indices
    sub = ds.X_OD[np.ix_(train_idx, train_idx)].astype(np.float64)
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
    bin_names = ['실제값 0', '실제값 1~100', '실제값 101~999', '실제값 1000이상']
    bin_counts = [int((bins == i).sum()) for i in range(4)]
    bin_ratios = [c / total_count for c in bin_counts]

    # 동일 행정동 내부 이동(대각) vs 동 간 이동(비대각) — train_indices 부분행렬 기준
    diag_values = np.diag(sub)
    off_diag_mask = ~np.eye(N, dtype=bool)
    offdiag_values = sub[off_diag_mask]

    same_dong_count = int(diag_values.size)
    same_dong_mean = float(diag_values.mean())
    diff_dong_count = int(offdiag_values.size)
    diff_dong_mean = float(offdiag_values.mean())

    # 기존 Weighted MSE(log1p target 기준, alpha 고정) 전체 loss에서 각 구간이 차지하는 비중.
    # 실제 예측오차는 학습 후에만 알 수 있으므로, "구간별 오차 크기가 균일하다"는 가정 하에
    # weight = 1 + alpha*log1p(target) 의 합으로 근사한다(가중치 총합 기준 비중).
    log_values = np.log1p(values)
    weights = 1.0 + args.alpha * log_values
    total_weight = weights.sum()
    bin_weight_share = [float(weights[bins == i].sum() / total_weight) for i in range(4)]

    # ---- CSV ----
    rows = [
        {'metric': 'total_od_count', 'value': total_count},
        {'metric': 'zero_ratio', 'value': zero_ratio},
        {'metric': 'positive_mean', 'value': pos_mean},
        {'metric': 'positive_median', 'value': pos_median},
        {'metric': 'positive_std', 'value': pos_std},
    ]
    for p in percentiles:
        rows.append({'metric': f'percentile_{p}_all', 'value': pct_all[p]})
    for p in percentiles:
        rows.append({'metric': f'percentile_{p}_positive_only', 'value': pct_pos[p]})
    for name, cnt, ratio, wshare in zip(bin_names, bin_counts, bin_ratios, bin_weight_share):
        rows.append({'metric': f'count[{name}]', 'value': cnt})
        rows.append({'metric': f'ratio[{name}]', 'value': ratio})
        rows.append({'metric': f'weighted_mse_loss_share[{name}]', 'value': wshare})
    rows.append({'metric': 'same_dong_count', 'value': same_dong_count})
    rows.append({'metric': 'same_dong_mean', 'value': same_dong_mean})
    rows.append({'metric': 'diff_dong_count', 'value': diff_dong_count})
    rows.append({'metric': 'diff_dong_mean', 'value': diff_dong_mean})

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.out_dir, 'target_distribution.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"Saved {csv_path}")

    # ---- MD ----
    md_lines = []
    md_lines.append("# Target 분포 분석 (SpatialODMAE1 학습 대상)")
    md_lines.append("")
    md_lines.append(f"- 분석 대상: train_indices × train_indices 부분행렬 (N={N}, 총 {total_count:,}개 OD 쌍)")
    md_lines.append(f"- 데이터: `dataset.py`의 `ODDataset(mode='train', isLogScale=False)` (real scale)")
    md_lines.append("")
    md_lines.append("## 기본 통계")
    md_lines.append(f"- 0의 비율: {zero_ratio:.4%}")
    md_lines.append(f"- 양수 OD 평균: {pos_mean:,.2f}")
    md_lines.append(f"- 양수 OD 중앙값: {pos_median:,.2f}")
    md_lines.append(f"- 양수 OD 표준편차: {pos_std:,.2f}")
    md_lines.append("")
    md_lines.append("## 분위수")
    md_lines.append("| 분위수 | 전체(0 포함) | 양수만 |")
    md_lines.append("|---|---|---|")
    for p in percentiles:
        md_lines.append(f"| {p}% | {pct_all[p]:,.2f} | {pct_pos[p]:,.2f} |")
    md_lines.append("")
    md_lines.append("## 구간별 표본 수 / 비율 / Weighted MSE 기여 비중(근사)")
    md_lines.append("| 구간 | 표본 수 | 비율 | Weighted MSE(alpha={:.1f}) 가중치 총합 기여 비중* |".format(args.alpha))
    md_lines.append("|---|---|---|---|")
    for name, cnt, ratio, wshare in zip(bin_names, bin_counts, bin_ratios, bin_weight_share):
        md_lines.append(f"| {name} | {cnt:,} | {ratio:.4%} | {wshare:.4%} |")
    md_lines.append("")
    md_lines.append("\\* 실제 예측오차는 학습 후에만 알 수 있으므로, 구간별 오차 크기가 균일하다는 가정 하에 "
                     "`weight = 1 + alpha*log1p(target)`의 총합 기준으로 근사한 값입니다. 실제 학습에서 "
                     "특정 구간의 오차가 더 크면 그 구간의 실질적 기여도는 이보다 더 커질 수 있습니다.")
    md_lines.append("")
    md_lines.append("## 동일 행정동 내부 vs 동 간 이동")
    md_lines.append("| 구분 | 표본 수 | 평균 |")
    md_lines.append("|---|---|---|")
    md_lines.append(f"| 동일 행정동 내부(대각) | {same_dong_count:,} | {same_dong_mean:,.2f} |")
    md_lines.append(f"| 서로 다른 행정동 간(비대각) | {diff_dong_count:,} | {diff_dong_mean:,.2f} |")
    md_lines.append("")

    md_path = os.path.join(args.out_dir, 'target_distribution.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    print(f"Saved {md_path}")

    # ---- 신규 loss 하이퍼파라미터 기본값(데이터 기반) ----
    # DualScaleMSELoss.global_scale: 배치별 폴백과 동일한 정의(RMS, 0 포함 전체)를 전역 데이터로 계산.
    global_scale_rms = float(np.sqrt(np.mean(values.astype(np.float64) ** 2)))
    # BinBalancedMSELoss.bin_freq: 위에서 계산한 구간별 비율 그대로.
    # TailAwareRelativeLoss.tau: 양수 OD의 median(0 근처 폭발 방지, "데이터 분포로 결정" 요구사항).
    tau_tail_aware = pos_median

    derived = {
        'dual_scale_mse': {'global_scale': global_scale_rms},
        'bin_balanced_mse': {'bin_freq': bin_ratios},
        'tail_aware_relative': {'tau': tau_tail_aware},
        'source': {
            'zero_ratio': zero_ratio,
            'positive_median': pos_median,
            'global_scale_rms_all': global_scale_rms,
            'bin_ratios': dict(zip(bin_names, bin_ratios)),
        },
    }
    hparam_path = os.path.join(args.out_dir, 'derived_loss_hparams.json')
    with open(hparam_path, 'w', encoding='utf-8') as f:
        json.dump(derived, f, ensure_ascii=False, indent=2)
    print(f"Saved {hparam_path}")


if __name__ == '__main__':
    main()
