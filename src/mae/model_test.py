import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from mae.dataset import ODDataset
from mae.models import SpatialODMAE


def cpc_score(y_true, y_pred):
    numerator   = 2 * np.sum(np.minimum(y_true, y_pred))
    denominator = np.sum(y_true) + np.sum(y_pred)
    if denominator == 0:
        return 0.0
    return numerator / denominator


def test_model(fold=1, model_path=None):
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 모델 경로 결정
    if model_path is None:
        model_path = os.path.join(current_dir, f'best_model_mae_fold_{fold}.pth')

    # ── 데이터 로드 ──────────────────────────────────────────────────────────
    test_dataset = ODDataset(mode='test')
    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps'  if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ── 모델 로드 ─────────────────────────────────────────────────────────────
    model = SpatialODMAE(num_nodes=test_dataset.num_nodes,
                          num_features=test_dataset.X_static.shape[1]).to(device)

    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found! Please train the model first.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    print(f"Loaded: {model_path}")

    # Dropout은 eval, BatchNorm은 train 유지 (twostage 방식 동일)
    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.eval()

    # ── 추론 ──────────────────────────────────────────────────────────────────
    all_y_true, all_y_pred = [], []
    pred_full = None

    with torch.no_grad():
        for batch in test_loader:
            x_static    = batch['X_static'].to(device)
            x_dist      = batch['X_dist'].to(device)
            mask        = batch['mask'].to(device)
            x_od_masked = batch['X_OD_masked'].to(device)
            y_od        = batch['y_OD'].to(device)

            pred    = model(x_static, x_od_masked, x_dist, mask)
            mask_2d = mask.unsqueeze(1) | mask.unsqueeze(2)

            # log 스케일 → real 스케일
            p_real = np.maximum(torch.expm1(pred[mask_2d]).cpu().numpy(), 0)
            y_real = torch.expm1(y_od[mask_2d]).cpu().numpy()

            all_y_true.append(y_real)
            all_y_pred.append(p_real)
            pred_full = torch.expm1(pred[0]).cpu().numpy()  # (N, N) full matrix

    all_y_true = np.concatenate(all_y_true)
    all_y_pred = np.concatenate(all_y_pred)

    # ── 지표 계산 ─────────────────────────────────────────────────────────────
    rmse = np.sqrt(np.mean((all_y_true - all_y_pred) ** 2))
    mae  = np.mean(np.abs(all_y_true - all_y_pred))
    cpc  = cpc_score(all_y_true, all_y_pred)
    corr = np.corrcoef(all_y_true, all_y_pred)[0, 1]

    print(f"\n=== Test Results (Fold {fold}) ===")
    print(f"RMSE : {rmse:.2f}")
    print(f"MAE  : {mae:.2f}")
    print(f"CPC  : {cpc:.4f}")
    print(f"Corr : {corr:.4f}")
    print(f"N    : {len(all_y_true):,}")

    # ── 시각화 (6-panel) ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # 1. Scatter (log scale)
    ax1 = fig.add_subplot(gs[0, 0])
    max_val = max(np.log1p(all_y_true).max(), np.log1p(all_y_pred).max())
    ax1.scatter(np.log1p(all_y_true), np.log1p(all_y_pred),
                alpha=0.15, s=2, c='steelblue')
    ax1.plot([0, max_val], [0, max_val], 'r--', lw=1.5, label='y=x')
    ax1.set_xlabel('True OD (log1p)')
    ax1.set_ylabel('Pred OD (log1p)')
    ax1.set_title(f'Scatter (log scale)\nCorr={corr:.3f}')
    ax1.legend()

    # 2. Residual Plot
    ax2 = fig.add_subplot(gs[0, 1])
    residuals = all_y_pred - all_y_true
    ax2.scatter(np.log1p(all_y_true), residuals, alpha=0.15, s=2, c='darkorange')
    ax2.axhline(0, color='r', linestyle='--', lw=1.5)
    ax2.set_xlabel('True OD (log1p)')
    ax2.set_ylabel('Residual (Pred - True)')
    ax2.set_title('Residual Plot')

    # 3. Residual Distribution
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(residuals, bins=80, color='slateblue', alpha=0.8,
             edgecolor='white', linewidth=0.3)
    ax3.axvline(0, color='r', linestyle='--')
    ax3.set_xlabel('Residual')
    ax3.set_ylabel('Count')
    ax3.set_title(f'Residual Dist (bias={residuals.mean():.1f})')

    # 4. 구간별 RMSE
    ax4 = fig.add_subplot(gs[1, 0])
    bins   = [0, 10, 50, 100, 300, 1000, np.inf]
    labels = ['0-10', '10-50', '50-100', '100-300', '300-1k', '1k+']
    bin_rmse, bin_cpc, bin_cnt = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        idx = (all_y_true >= lo) & (all_y_true < hi)
        if idx.sum() == 0:
            bin_rmse.append(0); bin_cpc.append(0); bin_cnt.append(0)
        else:
            bin_rmse.append(np.sqrt(np.mean((all_y_true[idx] - all_y_pred[idx])**2)))
            bin_cpc.append(cpc_score(all_y_true[idx], all_y_pred[idx]))
            bin_cnt.append(idx.sum())
    bars = ax4.bar(labels, bin_rmse, color='tomato', alpha=0.85)
    ax4.set_xlabel('True OD Range')
    ax4.set_ylabel('RMSE')
    ax4.set_title('RMSE by True OD Range')
    for bar, cnt in zip(bars, bin_cnt):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'n={cnt}', ha='center', va='bottom', fontsize=7)

    # 5. 구간별 CPC
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.bar(labels, bin_cpc, color='mediumseagreen', alpha=0.85)
    ax5.set_ylim(0, 1)
    ax5.set_xlabel('True OD Range')
    ax5.set_ylabel('CPC')
    ax5.set_title('CPC by True OD Range')

    # 6. 예측 OD 히트맵 (test 도시 행/열만)
    ax6 = fig.add_subplot(gs[1, 2])
    test_idx = test_dataset.test_indices[:20]   # 최대 20개 도시만 표시
    pred_sub = np.maximum(pred_full[np.ix_(test_idx, test_idx)], 0)
    im = ax6.imshow(np.log1p(pred_sub), aspect='auto', cmap='YlOrRd')
    ax6.set_title('Pred OD Heatmap\n(Test cities, log1p)')
    ax6.set_xlabel('Dest city index')
    ax6.set_ylabel('Origin city index')
    plt.colorbar(im, ax=ax6, fraction=0.046, pad=0.04)

    fig.suptitle(
        f'SpatialODMAE Test Results (Fold {fold})\n'
        f'RMSE={rmse:.2f}  MAE={mae:.2f}  CPC={cpc:.4f}  Corr={corr:.4f}',
        fontsize=13, fontweight='bold'
    )

    save_path = os.path.join(current_dir, f'test_analysis_fold_{fold}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved -> {save_path}")

    # ── Full OD Matrix CSV 저장 ───────────────────────────────────────────────
    try:
        import pandas as pd
        dong_path = os.path.join(current_dir, '..', '..', 'dataset', 'raw', 'OD_dong_list.xlsx')
        dong_df   = pd.read_excel(dong_path)
        dongs     = dong_df['dong_code'].values
        df_pred   = pd.DataFrame(np.maximum(pred_full, 0), index=dongs, columns=dongs)
        csv_path  = os.path.join(current_dir, f'predicted_OD_matrix_fold_{fold}.csv')
        df_pred.to_csv(csv_path)
        print(f"Full OD matrix saved -> {csv_path}")
    except Exception as e:
        print(f"(CSV 저장 스킵: {e})")

    return rmse, cpc


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=1, help='사용할 fold 번호')
    parser.add_argument('--model_path', type=str, default=None, help='가중치 경로 직접 지정 (선택)')
    args = parser.parse_args()

    test_model(fold=args.fold, model_path=args.model_path)