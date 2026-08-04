import os
os.environ['KMP_DUPLICATE_OK'] = 'True'
import numpy as np
import argparse, sys, torch, warnings, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataset import ODDataset
from collections import defaultdict
from model import DoublyConstrainedGravityModel
os.path.dirname(os.path.abspath(__file__))
from evaluation.fixed_eval_utils import apply_merge_events
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')


class TeeLogger:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, message):
        for stream in self.streams:
            stream.write(message)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def setup_logging(args):
    log_dir = os.path.abspath(args.log_dir)
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(
        log_dir,
        f"gravity_{args.model_type}_{args.imputation}_{timestamp}.log",
    )
    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = TeeLogger(sys.__stdout__, log_file)
    sys.stderr = TeeLogger(sys.__stderr__, log_file)
    print(f"[LOG] saving console output to {log_path}")
    return log_file, log_path


def format_minutes(seconds):
    return f"{seconds / 60:.1f}분"


# ===== 병렬 평가 worker (ThreadPoolExecutor용 — GIL 해제되는 numpy/LGBM은 스레드로 진짜 병렬) =====
def _eval_one_sample(args):
    """(model, base_data, ...) 받아 단일 샘플 평가 → dict 반환"""
    model, base_data, useLog, year_label, city_name, task, split_name, mask_indices, merge_events, imputation_values = args
    
    try:
        # validation에서는 test 동 정보를 입력에서 숨겨야 치팅이 아니다.
        # test에서는 test 동 자체가 평가 대상이므로 전체 feature를 hide하면 안 된다.
        # test 동에는 apply_merge_events 내부에서 mask_indices 기준 4개 컬럼 마스킹만 적용된다.
        holdout_indices = base_data['test_indices'] if split_name == 'val' else []
        sample = apply_merge_events(
            base_data,
            mask_indices,
            merge_events,
            hide_indices=holdout_indices,
            imputation_values=imputation_values,
        )
        
        # gravity 모델은 indicator 없는 raw_static(18 feature)으로 학습됨.
        # base_data의 X_static_raw는 mae-year dataset 기준으로 끝에
        # is_masked/is_merged indicator 2개가 붙어있을 수 있으므로 제거.
        n_lgbm_features = getattr(model.model_O, 'n_features_in_', None)
        x_raw = sample['X_static_raw'].float().numpy()
        if n_lgbm_features is not None and x_raw.shape[1] != n_lgbm_features:
            x_raw = x_raw[:, :n_lgbm_features]
        O_pred, D_pred = model.predict_O_D(x_raw, useLog)
        
        dist_matrix = sample['X_dist'].float().numpy()
        dist_no_diag = dist_matrix.copy()
        np.fill_diagonal(dist_no_diag, np.inf)
        intrazonal_dist = dist_no_diag.min(axis=1) / 2.0
        dist_matrix_modified = dist_matrix.copy()
        np.fill_diagonal(dist_matrix_modified, intrazonal_dist)
        
        T_pred = model.apply_ipf(O_pred, D_pred, dist_matrix_modified)
        
        y_od = sample['y_OD_raw'].numpy()
        eval_indices = np.array(mask_indices)
        N = y_od.shape[0]
        
        eval_mask_2d = np.zeros((N, N), dtype=bool)
        eval_mask_2d[:, eval_indices] = True
        eval_mask_2d[eval_indices, :] = True
        
        y_od_eval = y_od[eval_mask_2d]
        y_pred_eval = T_pred[eval_mask_2d]
        
        rmse_eval = np.sqrt(np.mean((y_od_eval - y_pred_eval) ** 2))
        num = 2 * np.sum(np.minimum(y_od_eval, y_pred_eval))
        den = np.sum(y_od_eval) + np.sum(y_pred_eval)
        cpc_eval = num / den if den > 0 else 0.0
        prmse_eval = rmse_eval / np.mean(y_od_eval) if np.mean(y_od_eval) > 0 else 0.0
        
        return {'year': year_label, 'city': city_name, 'task': task,
                        'rmse': rmse_eval, 'cpc': cpc_eval, 'prmse': prmse_eval,
                        'split': split_name,
                        'y_od_eval': y_od_eval,     # eval 대상 셀만 (1D) - 시각화용
                        'y_pred_eval': y_pred_eval, # eval 대상 셀만 (1D) - 시각화용
                        'T_pred': T_pred}           # 메모리 절약: evaluate_and_report에서 첫 결과 외 None으로 교체됨
    except Exception as e:
        print(f"  [WARN] 샘플 실패 ({city_name} task={task}): {e}")
        return None

def evaluate_and_report(base_data, val_meta, useRaw, model, useLog, year_label, split_name, imputation_values=None, n_workers=4):
    """ThreadPoolExecutor로 샘플병 병렬 평가"""
    
    job_args = []
    for task in [0, 1, 2, 3, 4]:
        for city_name, val_meta_task_list in val_meta.items():
            for meta in val_meta_task_list[task]:
                job_args.append((
                    model, base_data, useLog,
                    year_label, city_name, task, split_name,
                    meta['mask_indices'], meta['merge_events'], imputation_values
                ))
    
    total = len(job_args)
    print(f"  [{year_label}/{split_name}] {total}개 샘플 병렬 평가 중 (threads={n_workers})...")
    
    results = [None] * total
    heatmap_record = None  # 히트맵용 T_pred를 가진 대표 샘플 1개만 보존
    start_time = time.time()
    progress_every = 30
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_eval_one_sample, args): i
            for i, args in enumerate(job_args)
        }
        for done_count, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            if result is not None:
                # 첫 번째 성공 결과에서만 T_pred 보존, 나머지는 None으로 교체해 메모리 절약
                if heatmap_record is None and result.get('T_pred') is not None:
                    heatmap_record = result
                else:
                    result['T_pred'] = None
            results[futures[future]] = result

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
    
    records = [r for r in results if r is not None]
    # heatmap_record를 records 첫 번째 요소에 다시 반영 (T_pred 보존)
    for r in records:
        if r is heatmap_record:
            break
    else:
        # records 리스트 중 heatmap_record가 있으면 T_pred 복원 (이미 포함됨)
        pass
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
    
def main():
    '''
        실행방법: python train_and_test.py --model_type [lgbm|trip_rate|cross_class|linear_regression] --imputation [zero|mean]
        model_type: 통행발생량 예측 모델 유형 선택
        imputation: 결측치(마스킹된 피처) 처리 방식 (zero: 0.0, mean: 학습 데이터 평균)
    '''
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='lgbm',
                        choices=['lgbm', 'trip_rate', 'cross_class', 'linear_regression'],
                        help='통행발생량 예측 모델 유형 선택')
    parser.add_argument('--imputation', type=str, default='zero',
                        choices=['zero', 'mean'],
                        help='결측치(마스킹된 피처) 처리 방식 (zero: 0.0, mean: 학습 데이터 평균)')
    parser.add_argument('--log_dir', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs'),
                        help='콘솔 출력 로그 저장 폴더')
    args = parser.parse_args()
    log_file, log_path = setup_logging(args)
    
    year_labels = ['2019', '2023']
    datasets = []
    imputation_values_by_year = {}
    X_static_train_list = []
    X_o_train_list, X_d_train_list = [], []
    
    for year in year_labels:
        dataset = ODDataset(year=year, imputation=args.imputation, use_raw_static=args.model_type in ('trip_rate', 'cross_class', 'linear_regression'))
        X_static_train_list.append(dataset.X_static_train)
        X_o_train_list.append(dataset.y_o[dataset.train_mask])
        X_d_train_list.append(dataset.y_d[dataset.train_mask])
        datasets.append(dataset)
        if args.imputation == 'mean':
            # fixed_eval의 base_data는 imputation별로 따로 저장되어 있지 않다.
            # 따라서 평가 샘플을 만들 때 마스킹된 4개 feature를 train 평균으로 채우도록
            # 연도별 scaled/raw 평균을 apply_merge_events에 넘긴다.
            imputation_values_by_year[year] = {
                'scaled': np.mean(dataset.X_static[dataset.train_mask], axis=0),
                'raw': np.mean(dataset.X_static_raw[dataset.train_mask], axis=0),
            }
        else:
            imputation_values_by_year[year] = None
    
    X_static_train = np.concatenate(X_static_train_list, axis=0)
    X_o_train = np.concatenate(X_o_train_list, axis=0)
    X_d_train = np.concatenate(X_d_train_list, axis=0)

    # 3. 모델 초기화 및 실행
    print(f"argument='{args.model_type}'")
    model = DoublyConstrainedGravityModel(beta=2.0, max_iter=100, tol=1e-4, generation_model_type=args.model_type)
    useLog = args.model_type in ('trip_rate', 'cross_class', 'linear_regression')
    useRaw = args.model_type in ('trip_rate', 'cross_class', 'linear_regression')
    
    # 4. LGBM 통합 학습
    print("Start Model Training on Combined Data...")
    model.fit_O_D(X_static_train, X_o_train, X_d_train, useLog)
    
    # 프로젝트 루트: src/gravity(경훈)/ -> src/ -> 루트
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    fixed_eval_dir = os.path.join(base_dir, "dataset", "fixed_eval")
    all_records = []
    year_records = {year: [] for year in year_labels}
    
    for year in year_labels:
        base_data_path = os.path.join(fixed_eval_dir, f"base_data_{year}.pt")
        if not os.path.exists(base_data_path):
            print(f"[SKIP] base_data_{year}.pt not found")
            continue
        base_data = torch.load(base_data_path, weights_only=False)
        
        for split_name in ['val', 'test']:
            meta_path = os.path.join(fixed_eval_dir, f"fixed_{split_name}_meta_{year}.pt")
            if not os.path.exists(meta_path):
                print(f"[SKIP] fixed_{split_name}_meta_{year}.pt not found")
                continue
            meta_dict = torch.load(meta_path, weights_only=False)
            
            print(f"\nEvaluating {year} / {split_name} ...")
            records = evaluate_and_report(
                base_data,
                meta_dict,
                useRaw,
                model,
                useLog,
                year,
                split_name,
                imputation_values=imputation_values_by_year[year],
            )
            all_records.extend(records)
            if split_name == 'test':
                year_records[year].extend(records)
            
    print("1. 절대 모델의 구조와 하이퍼파라미터를 test가 좋아지는 방향으로 하지 말 것. -> val이 좋아지는 방향으로 조정")
    print("2. 모델의 성능이 좋아지는 게 목적이 아니라 기존 방식을 잘 재현하는게 목적이야.")
    print("\n===연도별===")
    summarize_results(all_records, ['year', 'split', 'task'], "task별 종합 성능 (모든 도시)")
    summarize_results(all_records, ['year', 'split', 'task', 'city'], "task/도시별 성능")
    summarize_results(all_records, ['year', 'split', 'city'], "도시별 성능 (모든 task)")

    print("\n===연도 종합===")
    summarize_results(all_records, ['split', 'task'], "task별 종합 성능 (모든 도시)")
    summarize_results(all_records, ['split', 'task', 'city'], "task/도시별 성능")
    summarize_results(all_records, ['split', 'city'], "도시별 성능 (모든 task)")
    
    print("\n===총 종합===")
    summarize_results(all_records, ['split'], "전체 종합 (연도/task/도시 모두 무관)")
    print(f"[LOG] saved to {log_path}")
    
    for year in year_labels:
        all_y_true, all_y_pred = [], []
        for record in year_records[year]:
            if record is not None:
                all_y_true.append(record['y_od_eval'])
                all_y_pred.append(record['y_pred_eval'])
                
        rmse = np.mean([r['rmse'] for r in year_records[year]])
        cpc = np.mean([r['cpc'] for r in year_records[year]])
        prmse = np.mean([r['prmse'] for r in year_records[year]])
        
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
        current_dir = os.path.dirname(os.path.abspath(__file__))
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        result_dir = os.path.join(BASE_DIR, '../result')
    
        save_path = os.path.join(result_dir, f'result_gravity_{year}.png')
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
            csv_path  = os.path.join(result_dir, f'predicted_OD_matrix_gravity_{year}.csv')
            # df_pred.to_csv(csv_path)
            print(f"Full OD matrix saved -> {csv_path}")
        except Exception as e:
            print(f"(CSV 저장 스킵: {e})")
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    log_file.close()

# CPC 점수 함수
def cpc_score(y_t, y_p):
    num = 2 * np.sum(np.minimum(y_t, y_p))
    den = np.sum(y_t) + np.sum(y_p)
    return num / den if den > 0 else 0.0

if __name__ == "__main__":
    main()
