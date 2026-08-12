import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from dataset import ODDataset
from models import ODMAE
from validation import evaluate_and_report, summarize_results

def cpc_score(y_true, y_pred):
    numerator   = 2 * np.sum(np.minimum(y_true, y_pred))
    denominator = np.sum(y_true) + np.sum(y_pred)
    if denominator == 0:
        return 0.0
    return numerator / denominator


def test_model(model_path=None, use_lgbm_self_loop=False, year='2023', mode = 'val'):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    best_model_dir = os.path.join(BASE_DIR, '../best_model')
    result_dir = os.path.join(BASE_DIR, '../result')
    fixed_eval_dir = os.path.join(BASE_DIR, '../dataset/fixed_eval')
    os.makedirs(result_dir, exist_ok=True)

    # === 1. 모델 경로 설정 ===
    if model_path is None:
        if not os.path.exists(best_model_dir):
            raise FileNotFoundError(f"E: {best_model_dir} 디렉토리가 없습니다.")
        
        pth_files = [f for f in os.listdir(best_model_dir) if f.endswith('.pth')]
        if not pth_files:
            raise FileNotFoundError(f"E: {best_model_dir} 내에 .pth 가중치 파일이 없습니다.")
            
        print("\n==================================")
        print("테스트할 가중치 파일을 선택하세요:")
        for idx, f in enumerate(pth_files):
            print(f"[{idx+1}] {f}")
        print("==================================")
        
        try:
            sel = int(input("번호 입력: ")) - 1
            if sel < 0 or sel >= len(pth_files):
                raise ValueError
            model_name = pth_files[sel]
            model_path = os.path.join(best_model_dir, model_name)
        except Exception:
            raise ValueError("E: 잘못된 입력.")
    else:
        model_name = os.path.basename(model_path)
        
    print("\n==================================")
    print("이 가중치는 Transformer를 사용한 원본 모델입니까, FFN Ablation 모델입니까?")
    print("[1] Transformer (원본, 기본값)")
    print("[2] FFN Ablation")
    print("==================================")
    trans_sel = input("번호 입력 (엔터시 기본값): ").strip()
    use_transformer = False if trans_sel == '2' else True
        
    model_base_name = os.path.splitext(model_name)[0]
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"E: {model_path} 가중치 파일이 없습니다.")

    # === 2. test 데이터셋 로드 ===
    dataset = ODDataset(year=year)
    base_data = torch.load(os.path.join(fixed_eval_dir, f'base_data_{year}.pt'), map_location='cpu', weights_only=False)
    meta_data = torch.load(os.path.join(fixed_eval_dir, f'fixed_{mode}_meta_{year}.pt'), map_location='cpu', weights_only=False)
    
    # === 3. 모델 로드 ===
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps'  if torch.backends.mps.is_available() else 'cpu')
    print(f"I: Using device: {device}")
    
    F = dataset.X_static.shape[1]
    
    model = ODMAE(num_features=F).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False), strict=False)
    print(f"I: Loaded: {model_path}")

    model.eval()

    # === 4. 평가 ===
    records = evaluate_and_report(
        base_data=base_data,
        val_meta=meta_data,
        model=model,
        year_label=year,
        split_name=mode,
        n_workers=1,
        device=device,
        use_lgbm_self_loop=use_lgbm_self_loop
    )
    print("\n=== Validation Results ===")
    summarize_results(records, ['task'], "연도별 task 요약")
    summarize_results(records, ['city'], "도시별 task 요약")
    summarize_results(records, [], "연도별 총 요약")
    
    rmse = np.mean([r['rmse'] for r in records])
    cpc = np.mean([r['cpc'] for r in records])
    prmse = np.mean([r['prmse'] for r in records])
    top20 = np.mean([r['top20_acc'] for r in records if 'top20_acc' in r])
    vol = np.mean([r['vol_ratio'] for r in records if 'vol_ratio' in r])
    cpc_s = np.mean([r['cpc_self'] for r in records if 'cpc_self' in r])
    cpc_e = np.mean([r['cpc_ext'] for r in records if 'cpc_ext' in r])

    print(f"  ➜ [Val] RMSE: {rmse:.2f} | CPC: {cpc:.4f} (Self:{cpc_s:.4f} Ext:{cpc_e:.4f}) | PRMSE: {prmse:.4f} | VolRatio: {vol:.4f} | Top20: {top20:.4f}")
    
    # === 5. 시각화 및 지표 계산 ===
    all_y_true, all_y_pred = [], []
    for record in records:
        if record is not None:
            all_y_true.append(record['y_od_eval'])
            all_y_pred.append(record['y_pred_eval'])
    
    all_y_true = np.concatenate(all_y_true)
    all_y_pred = np.concatenate(all_y_pred)

    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # 5.1. Scatter (log scale)
    ax1 = fig.add_subplot(gs[0, 0])
    max_val = max(np.log1p(all_y_true).max(), np.log1p(all_y_pred).max())
    ax1.scatter(np.log1p(all_y_true), np.log1p(all_y_pred),
                alpha=0.15, s=2, c='steelblue')
    ax1.plot([0, max_val], [0, max_val], 'r--', lw=1.5, label='y=x')
    ax1.set_xlabel('True OD (log1p)')
    ax1.set_ylabel('Pred OD (log1p)')
    ax1.set_title(f'Scatter (log scale)')
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

    # 6. 예측 OD 히트맵 (대표 샘플 1개)
    ax6 = fig.add_subplot(gs[1, 2])
    rep_record = next((r for r in records if r is not None and r.get('T_pred') is not None), None)
    if rep_record is not None:
        T_rep = np.maximum(rep_record['T_pred'], 0)
        mask_idx = np.array(list(set(np.where(T_rep.sum(axis=1) > 0)[0])))[:20]
        if len(mask_idx) > 1:
            pred_sub = T_rep[np.ix_(mask_idx, mask_idx)]
        else:
            pred_sub = T_rep[:20, :20]
        im = ax6.imshow(np.log1p(pred_sub), aspect='auto', cmap='YlOrRd')
        ax6.set_title(f'Pred OD Heatmap\n({rep_record["city"]} task={rep_record["task"]}, log1p)')
        ax6.set_xlabel('Dest city index')
        ax6.set_ylabel('Origin city index')
        plt.colorbar(im, ax=ax6, fraction=0.046, pad=0.04)
    else:
        ax6.set_visible(False)

    fig.suptitle(
        f'SpatialODMAE Test Results\n'
        f'RMSE={rmse:.2f}  CPC={cpc:.4f}  %RMSE={prmse:.4f}',
        fontsize=13, fontweight='bold'
    )

    save_path = os.path.join(result_dir, f'result_{model_base_name}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved -> {save_path}")

    # ── Full OD Matrix CSV 저장 ───────────────────────────────────────────────
    try:
        import pandas as pd
        dong_path = os.path.join(current_dir, '..', '..', 'dataset', 'raw', 'OD_dong_list.xlsx')
        dong_df   = pd.read_excel(dong_path)
        dongs     = dong_df['dong_code'].values
        # df_pred   = pd.DataFrame(np.maximum(pred_full, 0), index=dongs, columns=dongs)
        csv_path  = os.path.join(result_dir, f'predicted_OD_matrix_{model_base_name}.csv')
        # df_pred.to_csv(csv_path)
        print(f"Full OD matrix saved -> {csv_path}")
    except Exception as e:
        print(f"(CSV 저장 스킵: {e})")

    return rmse, cpc


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None, help='가중치 경로 직접 지정 (선택)')
    parser.add_argument('--use_lgbm_self_loop', type=str, default='False')
    parser.add_argument('--year', type=str, default='2023', help='평가할 연도 (기본: 2023)')
    parser.add_argument('--mode', type=str, default='val', help='평가 모드 (기본: test)', choices=['test', 'val'])
    args = parser.parse_args()
    
    use_lgbm_self_loop_bool = str(args.use_lgbm_self_loop).lower() in ("yes", "true", "t", "1")

    test_model(
        model_path=args.model_path, 
        use_lgbm_self_loop=use_lgbm_self_loop_bool,
        year = args.year,
        mode = args.mode
    )