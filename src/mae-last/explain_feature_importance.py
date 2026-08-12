"""
Static Feature + Distance 중요도 분석

[방법론]
- Panel 1: Input × Gradient (국소 선형 민감도) — feature 간 상대 순위 비교용
- Panel 2: Mean-Replacement Perturbation — feature + distance 동일 척도 비교용
  · 각 feature: 해당 열을 노드 평균으로 교체 → 예측 변화량
  · distance:   전체 행렬을 최대값(5.5)으로 교체 → 예측 변화량
  → 두 값 모두 "제거했을 때 예측이 얼마나 흔들리는가" 로 같은 척도

실행법:
    cd /Users/implement/KT/KTDB/src/mae-year
    python explain_feature_importance.py \\
        --ckpt /Users/implement/KT/KTDB/best_model/mae:hybrid_cpc-2023.pth \\
        --year 2023
"""

import os, sys, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# macOS 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.fixed_eval_utils import apply_merge_events


# ──────────────────────────────────────────────────────────
# 타겟 스코어: 마스킹 노드 관련 OD 합
# ──────────────────────────────────────────────────────────
def _target_score(pred, m):
    return pred[0, m, :].sum() + pred[0, :, m].sum()


# ──────────────────────────────────────────────────────────
# Panel 1: Input × Gradient (feature only, 로컬 민감도)
# ──────────────────────────────────────────────────────────
def grad_importance(model, x_static, x_od, x_dist, A, mask, anm, m):
    x_static = x_static.clone().requires_grad_(True)
    with torch.enable_grad():
        pred = model(x_static, x_od, x_dist, A, mask, anm)
        _target_score(pred, m).backward()

    if x_static.grad is None:
        F = x_static.shape[-1]
        return np.zeros(F), np.zeros(F), np.zeros(F)

    grad_s = x_static.grad[0].detach().cpu().numpy()   # (N, F)
    val_s  = x_static[0].detach().cpu().numpy()         # (N, F)
    imp    = np.abs(grad_s * val_s)                      # (N, F)

    mask_np = m.cpu().numpy()
    feat_all      = imp.mean(axis=0)
    feat_masked   = imp[mask_np].mean(axis=0) if mask_np.any()   else feat_all
    feat_unmasked = imp[~mask_np].mean(axis=0) if (~mask_np).any() else feat_all
    return feat_all, feat_masked, feat_unmasked


# ──────────────────────────────────────────────────────────
# Panel 2: Mean-Replacement Perturbation (feature + distance)
# ──────────────────────────────────────────────────────────
def perturb_importance(model, x_static, x_od, x_dist, A, mask, anm, m):
    F = x_static.shape[-1]
    DIST_MAX = 5.5  # log1p 스케일 최대값

    with torch.no_grad():
        pred_orig  = model(x_static, x_od, x_dist, A, mask, anm)
        score_orig = _target_score(pred_orig, m).item()

        # Feature: 각 열을 노드 평균으로 교체
        feat_imp = np.zeros(F)
        for f in range(F):
            x_s_p = x_static.clone()
            col_mean = x_static[0, :, f].mean()
            x_s_p[0, :, f] = col_mean
            pred_p = model(x_s_p, x_od, x_dist, A, mask, anm)
            feat_imp[f] = abs(score_orig - _target_score(pred_p, m).item())

        # Distance: 전체 행렬을 최대값으로 교체
        x_dist_p = torch.full_like(x_dist, DIST_MAX)
        pred_dp  = model(x_static, x_od, x_dist_p, A, mask, anm)
        dist_imp = abs(score_orig - _target_score(pred_dp, m).item())

    return feat_imp, dist_imp


# ──────────────────────────────────────────────────────────
# 단일 샘플 → 두 방법론의 중요도 반환
# ──────────────────────────────────────────────────────────
def importance_one_sample(model, sample, device):
    x_static = sample['X_static'].unsqueeze(0).to(device).float()
    x_dist   = sample['X_dist'].unsqueeze(0).to(device).float()
    x_od     = sample['X_OD_masked'].unsqueeze(0).to(device).float()
    A        = sample['A_spatial'].unsqueeze(0).to(device).float()
    mask     = sample['mask'].unsqueeze(0).to(device)
    anm      = sample['active_node_mask'].unsqueeze(0).to(device)
    m        = mask[0]

    grad_all, grad_masked, grad_unmasked = grad_importance(
        model, x_static, x_od, x_dist, A, mask, anm, m)
    perturb_feat, perturb_dist = perturb_importance(
        model, x_static, x_od, x_dist, A, mask, anm, m)

    return grad_all, grad_masked, grad_unmasked, perturb_feat, perturb_dist


# ──────────────────────────────────────────────────────────
# 시각화: 두 패널 나란히
# ──────────────────────────────────────────────────────────
def plot_two_panel(feat_names,
                   grad_all, grad_masked, grad_unmasked,   # (F,) each
                   perturb_feat, perturb_dist,              # (F,), scalar
                   n_samples, out_path, title_suffix=''):
    F = len(feat_names)

    # ── 정렬 순서 각각 독립 ──────────────────────────────
    # Panel 1: Gradient 기준 내림차순 (feature만, distance 제외)
    order_g = np.argsort(grad_all)[::-1]
    # Panel 2: Perturbation 기준 내림차순 (feature + distance 포함)
    perturb_all = np.append(perturb_feat, perturb_dist)
    all_names_p = feat_names + ['[거리]']
    order_p = np.argsort(perturb_all)[::-1]

    # ── 정규화 (각 패널 내부에서만) ─────────────────────
    s_g = grad_all.sum()
    g_all_n = grad_all / s_g if s_g > 0 else grad_all
    g_mask_n = grad_masked / s_g if s_g > 0 else grad_masked
    g_obs_n  = grad_unmasked / s_g if s_g > 0 else grad_unmasked

    s_p = perturb_all.sum()
    p_all_n = perturb_all / s_p if s_p > 0 else perturb_all

    # ── Figure 구성 ──────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(16, max(6, F * 0.40 + 2)),
        gridspec_kw={'width_ratios': [1, 1]}
    )
    h = 0.22

    # ─── Panel 1: Gradient ───────────────────────────────
    y1 = np.arange(F)
    s_names_g = [feat_names[i] for i in order_g]
    ax1.barh(y1 + h, g_all_n[order_g],  h, color='#9E9E9E', alpha=0.55, label='전체 평균')
    ax1.barh(y1,     g_mask_n[order_g], h, color='#E53935', alpha=0.85, label='마스킹 노드')
    ax1.barh(y1 - h, g_obs_n[order_g],  h, color='#1E88E5', alpha=0.85, label='관측 노드')
    ax1.set_yticks(y1)
    ax1.set_yticklabels(s_names_g, fontsize=8.5)
    ax1.set_xlabel('Normalized |input × gradient|', fontsize=9)
    ax1.set_title('Panel 1: Input × Gradient\n(Feature 내부 순위 비교용)\n⚠ Distance 제외 — 척도 달라 합산 불가', fontsize=9)
    ax1.legend(fontsize=8, loc='lower right')
    ax1.grid(axis='x', alpha=0.3)

    # ─── Panel 2: Perturbation ───────────────────────────
    s_names_p = [all_names_p[i] for i in order_p]
    n_bars = len(s_names_p)
    y2 = np.arange(n_bars)
    colors_p = ['#4CAF50' if n == '[거리]' else '#7B1FA2' for n in s_names_p]
    ax2.barh(y2, p_all_n[order_p], 0.5, color=colors_p, alpha=0.80)
    ax2.set_yticks(y2)
    ax2.set_yticklabels(s_names_p, fontsize=8.5)
    ax2.set_xlabel('Normalized Perturbation Impact\n(예측 변화량 / 전체 합)', fontsize=9)
    ax2.set_title('Panel 2: Mean-Replacement Perturbation\n(Feature + Distance 동일 척도 비교)\n🟢 [거리] = 최대값 교체  🟣 feature = 평균값 교체', fontsize=9)

    # [거리] 막대에 값 표시
    for i, (name, val) in enumerate(zip(s_names_p, p_all_n[order_p])):
        if name == '[거리]':
            ax2.text(val + 0.002, i, f'{val:.3f}', va='center', fontsize=8,
                     color='#2E7D32', fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)

    fig.suptitle(
        f'Static Feature + Distance 중요도  (n={n_samples} 샘플){title_suffix}',
        fontsize=11, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → 저장: {out_path}")


# ──────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────
def main(args):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fixed_eval = os.path.join(BASE_DIR, '../dataset/fixed_eval')
    result_dir = os.path.join(BASE_DIR, '../result')
    os.makedirs(result_dir, exist_ok=True)

    from dataset import ODDataset
    from models import ODMAE
    import pandas as pd
    from config import STATIC_DATA_23_PATH, STATIC_DATA_19_PATH

    dataset = ODDataset(year=args.year)
    F = dataset.X_static.shape[1]

    # feature 이름 재구성
    static_path = STATIC_DATA_23_PATH if args.year == '2023' else STATIC_DATA_19_PATH
    static_df   = pd.read_csv(static_path)
    feat_names  = sorted([c for c in static_df.columns if c not in ['dong_code', 'dong_name']])
    feat_names += ['is_masked', 'is_merged']
    assert len(feat_names) == F, f"feature 개수 불일치: {len(feat_names)} vs {F}"

    model = ODMAE(num_features=F, d_model=128, nhead=8, num_layers=4).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"✅ Model loaded: {args.ckpt}")

    base_data = torch.load(
        os.path.join(fixed_eval, f'base_data_{args.year}.pt'),
        map_location='cpu', weights_only=False)
    val_meta = torch.load(
        os.path.join(fixed_eval, f'fixed_val_meta_{args.year}.pt'),
        map_location='cpu', weights_only=False)

    cities = [c.strip() for c in args.cities.split(',')]
    tasks  = [int(t) for t in args.tasks.split(',')]

    acc_grad_all, acc_grad_masked, acc_grad_unmasked = [], [], []
    acc_perturb_feat, acc_perturb_dist = [], []

    for city in cities:
        if city not in val_meta:
            print(f"  ⚠ '{city}' not found"); continue
        for task in tasks:
            for meta in val_meta[city][task][:args.max_per_task]:
                sample = apply_merge_events(
                    base_data, meta['mask_indices'], meta['merge_events'],
                    hide_indices=base_data['test_indices'])
                try:
                    ga, gm, gu, pf, pd_ = importance_one_sample(model, sample, device)
                    acc_grad_all.append(ga);     acc_grad_masked.append(gm)
                    acc_grad_unmasked.append(gu); acc_perturb_feat.append(pf)
                    acc_perturb_dist.append(pd_)
                    print(f"  [{city}/task={task}] 완료  (perturb_dist={pd_:.1f})")
                except Exception as e:
                    import traceback; traceback.print_exc()
                    print(f"  ⚠ [{city}/task={task}] 실패: {e}")

    if not acc_grad_all:
        print("수집된 샘플이 없습니다."); return

    n = len(acc_grad_all)
    avg_ga  = np.stack(acc_grad_all).mean(0)
    avg_gm  = np.stack(acc_grad_masked).mean(0)
    avg_gu  = np.stack(acc_grad_unmasked).mean(0)
    avg_pf  = np.stack(acc_perturb_feat).mean(0)
    avg_pd  = float(np.mean(acc_perturb_dist))

    # ── 콘솔 요약 ─────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"Panel 1 - Gradient  (n={n})  ← Feature 내부 순위만 비교하세요")
    print(f"{'='*65}")
    order_g = np.argsort(avg_ga)[::-1]
    for i in order_g:
        pct = avg_ga[i] / avg_ga.sum() * 100
        print(f"  {feat_names[i]:<28} {pct:>6.2f}%")

    print(f"\n{'='*65}")
    print(f"Panel 2 - Perturbation  (n={n})  ← Feature + Distance 동일 척도")
    print(f"{'='*65}")
    pa = np.append(avg_pf, avg_pd)
    pn = list(feat_names) + ['[거리]']
    order_p = np.argsort(pa)[::-1]
    for i in order_p:
        pct = pa[i] / pa.sum() * 100
        print(f"  {pn[i]:<28} {pct:>6.2f}%")

    suffix = f'\ncities={cities}  tasks={tasks}'
    plot_two_panel(
        feat_names, avg_ga, avg_gm, avg_gu, avg_pf, avg_pd,
        n_samples=n,
        out_path=os.path.join(result_dir, 'feature_importance.png'),
        title_suffix=suffix
    )
    print(f"\n✅ 완료")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',  type=str, required=True)
    parser.add_argument('--year',  type=str, default='2023', choices=['2019', '2023'])
    parser.add_argument('--cities', type=str, default='동탄,위례,검단')
    parser.add_argument('--tasks',  type=str, default='0,1,2,3,4')
    parser.add_argument('--max_per_task', type=int, default=3,
                        help='city×task당 최대 샘플 수')
    args = parser.parse_args()
    main(args)
