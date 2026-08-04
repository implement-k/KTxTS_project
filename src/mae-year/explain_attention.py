"""
Attention Head Analysis for ODMAE — Multi-city / Multi-task 버전
대각선 제외, 여러 도시 × 여러 task에 걸쳐 평균을 내어 패턴의 일반성을 확인.

실행법:
    cd /Users/implement/KT/KTDB/src/mae-year
    python explain_attention.py \
        --ckpt /Users/implement/KT/KTDB/best_model/mae:hybrid_cpc-2023.pth \
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
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
import seaborn as sns
from sklearn.preprocessing import normalize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.fixed_eval_utils import apply_merge_events

# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────
def cosine_sim_matrix(X):
    """(N, F) → (N, N) pairwise cosine similarity"""
    X_norm = normalize(X, norm='l2', axis=1)
    return (X_norm @ X_norm.T).astype(np.float32)

def same_city_matrix(mask_idx, N):
    """마스킹된 노드들끼리 = 1, 나머지 = 0"""
    m = np.zeros((N, N), dtype=np.float32)
    m[np.ix_(mask_idx, mask_idx)] = 1.0
    return m

def corr_no_diag(a, b):
    """대각선 제외한 Pearson 상관계수"""
    N = a.shape[0]
    assert a.shape == b.shape == (N, N)
    off_diag = ~np.eye(N, dtype=bool)
    valid = off_diag & np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 10:
        return float('nan')
    return float(np.corrcoef(a[valid], b[valid])[0, 1])


# ──────────────────────────────────────────────────────────
# Attention weight 추출 (Monkey-patch)
# ──────────────────────────────────────────────────────────
def extract_attn_weights(model, x_static, x_od_masked, x_dist, A_spatial, mask, active_node_mask):
    """각 TransformerEncoderLayer에서 per-head attention weight 추출.
    반환: {layer_idx: (nhead, N, N) numpy array}
    """
    captured = {}
    original_forwards = {}

    for layer_idx, layer in enumerate(model.transformer.layers):
        original_forwards[layer_idx] = layer.self_attn.forward

        def _make_patched(idx, orig_fwd):
            def patched(query, key, value, *args, **kwargs):
                kwargs['need_weights'] = True
                kwargs['average_attn_weights'] = False  # per-head
                out, w = orig_fwd(query, key, value, *args, **kwargs)
                if w is not None:
                    captured[idx] = w.detach().cpu()  # (B, nhead, N, N)
                return out, w
            return patched

        layer.self_attn.forward = _make_patched(layer_idx, original_forwards[layer_idx])

    try:
        with torch.no_grad():
            _ = model(x_static, x_od_masked, x_dist, A_spatial, mask, active_node_mask)
    finally:
        for layer_idx, layer in enumerate(model.transformer.layers):
            layer.self_attn.forward = original_forwards[layer_idx]

    # batch dim 제거 → {layer_idx: (nhead, N, N)}
    return {k: v[0].numpy() for k, v in captured.items()}


# ──────────────────────────────────────────────────────────
# 단일 샘플 분석 → corr dict 반환
# ──────────────────────────────────────────────────────────
def analyze_one_sample(model, base_data, mask_indices, merge_events, device, split_name='val'):
    holdout = base_data['test_indices'] if split_name == 'val' else []
    sample = apply_merge_events(base_data, mask_indices, merge_events, hide_indices=holdout)

    N = sample['X_static'].shape[0]
    x_static      = sample['X_static'].unsqueeze(0).to(device)
    x_od_masked   = sample['X_OD_masked'].unsqueeze(0).to(device)
    x_dist        = sample['X_dist'].unsqueeze(0).to(device)
    A_spatial     = sample['A_spatial'].unsqueeze(0).to(device)
    mask          = sample['mask'].unsqueeze(0).to(device)
    active_node_mask = sample['active_node_mask'].unsqueeze(0).to(device)

    captured = extract_attn_weights(model, x_static, x_od_masked, x_dist, A_spatial, mask, active_node_mask)
    if not captured:
        return None

    # 가설 행렬 구성
    dist_np = x_dist[0].cpu().numpy()
    cos_sim = cosine_sim_matrix(sample['X_static_raw'].numpy())

    od_np = sample['X_OD_masked'].numpy()
    od_scale = od_np.mean(axis=1) + od_np.mean(axis=0)
    od_scale_sim = -np.abs(od_scale[:, None] - od_scale[None, :])

    city_mat = same_city_matrix(np.array(mask_indices), N)

    hypotheses = {
        'distance':           dist_np,
        'feat_similarity':    cos_sim,
        'od_scale_sim':       od_scale_sim,
        'same_city':          city_mat,
    }

    # (layer, head) → {hyp: corr}
    corrs = {}
    for layer_idx, attn_w in captured.items():      # attn_w: (nhead, N, N)
        for head_idx in range(attn_w.shape[0]):
            attn_h = attn_w[head_idx]               # (N, N)
            corrs[(layer_idx, head_idx)] = {
                hyp: corr_no_diag(attn_h, hyp_mat)
                for hyp, hyp_mat in hypotheses.items()
            }
    return corrs


# ──────────────────────────────────────────────────────────
# 시각화
# ──────────────────────────────────────────────────────────
def plot_heatmap(mat, hyp_names, nhead, title, out_path, annot_mat=None):
    fig, ax = plt.subplots(figsize=(len(hyp_names) * 2.2 + 0.5, nhead * 0.6 + 1.5))
    annot = np.array([[f"{mat[h,j]:.3f}" for j in range(len(hyp_names))]
                      for h in range(nhead)])
    if annot_mat is not None:
        annot = np.array([[f"{mat[h,j]:.3f}\n±{annot_mat[h,j]:.3f}"
                           for j in range(len(hyp_names))]
                          for h in range(nhead)])
    sns.heatmap(
        mat, annot=annot, fmt='', cmap='coolwarm', center=0,
        xticklabels=hyp_names,
        yticklabels=[f'Head {h}' for h in range(nhead)],
        ax=ax, vmin=-0.6, vmax=0.6, linewidths=0.5
    )
    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → 저장: {out_path}")


# ──────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────
def main(args):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fixed_eval_dir = os.path.join(BASE_DIR, '../dataset/fixed_eval')
    result_dir = os.path.join(BASE_DIR, '../result')
    os.makedirs(result_dir, exist_ok=True)

    # 모델 로드
    from dataset import ODDataset
    from models import ODMAE

    dataset = ODDataset(year=args.year)
    F = dataset.X_static.shape[1]
    model = ODMAE(num_features=F, d_model=128, nhead=8, num_layers=4).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"✅ Model loaded: {args.ckpt}")

    # 데이터 로드
    base_data = torch.load(
        os.path.join(fixed_eval_dir, f'base_data_{args.year}.pt'),
        map_location='cpu', weights_only=False
    )
    val_meta = torch.load(
        os.path.join(fixed_eval_dir, f'fixed_val_meta_{args.year}.pt'),
        map_location='cpu', weights_only=False
    )

    cities = [c.strip() for c in args.cities.split(',')]
    tasks  = [int(t) for t in args.tasks.split(',')]
    hyp_names = ['distance', 'feat_similarity', 'od_scale_sim', 'same_city']
    nhead = 8

    # 결과 수집: {layer_idx → list of (nhead, n_hyp) corr matrices}
    from collections import defaultdict
    all_corrs = defaultdict(list)   # layer_idx → list of (nhead, n_hyp)
    sample_log = []

    for city in cities:
        if city not in val_meta:
            print(f"  ⚠ '{city}' not in val_meta (available: {list(val_meta.keys())})")
            continue
        for task in tasks:
            meta_list = val_meta[city][task]
            # task당 최대 args.max_per_task개 샘플만 사용
            for meta in meta_list[:args.max_per_task]:
                mask_indices = meta['mask_indices']
                merge_events = meta['merge_events']

                corrs = analyze_one_sample(
                    model, base_data, mask_indices, merge_events, device
                )
                if corrs is None:
                    continue

                n_layers = max(k[0] for k in corrs) + 1
                for layer_idx in range(n_layers):
                    mat = np.full((nhead, len(hyp_names)), float('nan'))
                    for head_idx in range(nhead):
                        key = (layer_idx, head_idx)
                        if key in corrs:
                            for j, hyp in enumerate(hyp_names):
                                mat[head_idx, j] = corrs[key].get(hyp, float('nan'))
                    all_corrs[layer_idx].append(mat)

                sample_log.append((city, task))

    n_samples = len(sample_log)
    print(f"\n총 {n_samples}개 샘플 수집: {set(sample_log)}")
    if n_samples == 0:
        print("분석할 샘플이 없습니다.")
        return

    # 레이어별 평균/표준편차
    n_layers = max(all_corrs.keys()) + 1
    avg_all_layers = np.zeros((nhead, len(hyp_names)))
    std_all_layers = np.zeros((nhead, len(hyp_names)))

    for layer_idx in range(n_layers):
        stack = np.stack(all_corrs[layer_idx], axis=0)   # (n_samples, nhead, n_hyp)
        avg  = np.nanmean(stack, axis=0)                  # (nhead, n_hyp)
        std  = np.nanstd(stack, axis=0)

        avg_all_layers += avg
        std_all_layers += std

        # 콘솔 출력
        print(f"\n{'='*65}")
        print(f"Layer {layer_idx}  (평균 / 표준편차,  n={n_samples}샘플)")
        print(f"{'='*65}")
        header = f"{'':8s}" + "".join(f"{h:>16s}" for h in hyp_names)
        print(header)
        for head_idx in range(nhead):
            row = f"Head {head_idx}  " + "".join(
                f"  {avg[head_idx, j]:+.3f}±{std[head_idx, j]:.3f}"
                for j in range(len(hyp_names))
            )
            print(row)

        plot_heatmap(
            avg, hyp_names, nhead,
            title=f'Attn Head Corr (Layer {layer_idx}) — avg over {n_samples} samples\n'
                  f'cities={cities}, tasks={tasks}  (대각선 제외)',
            out_path=os.path.join(result_dir, f'attn_corr_layer{layer_idx}_avg.png'),
            annot_mat=std
        )

    # 레이어 평균 총 요약
    avg_all_layers /= n_layers
    std_all_layers /= n_layers
    print(f"\n{'='*65}")
    print(f"▶ 전체 레이어 평균 (n_layers={n_layers}, n_samples={n_samples})")
    print(f"{'='*65}")
    header = f"{'':8s}" + "".join(f"{h:>16s}" for h in hyp_names)
    print(header)
    for head_idx in range(nhead):
        row = f"Head {head_idx}  " + "".join(
            f"  {avg_all_layers[head_idx, j]:+.3f}±{std_all_layers[head_idx, j]:.3f}"
            for j in range(len(hyp_names))
        )
        print(row)

    plot_heatmap(
        avg_all_layers, hyp_names, nhead,
        title=f'Attn Head Corr — ALL layers avg\n'
              f'cities={cities}, tasks={tasks}, n={n_samples} samples (대각선 제외)',
        out_path=os.path.join(result_dir, 'attn_corr_all_layers_avg.png'),
        annot_mat=std_all_layers
    )

    print(f"\n✅ 완료. 결과 이미지: {result_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',  type=str, required=True)
    parser.add_argument('--year',  type=str, default='2023', choices=['2019', '2023'])
    parser.add_argument('--cities', type=str, default='동탄,위례,검단',
                        help='콤마로 구분된 도시 목록')
    parser.add_argument('--tasks',  type=str, default='0,1,2,3,4',
                        help='콤마로 구분된 task 번호')
    parser.add_argument('--max_per_task', type=int, default=3,
                        help='city×task당 사용할 최대 샘플 수')
    args = parser.parse_args()
    main(args)
