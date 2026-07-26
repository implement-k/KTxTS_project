import os
os.environ['KMP_DUPLICATE_OK'] = 'True'
import numpy as np
import argparse, sys, torch, warnings
from concurrent.futures import ThreadPoolExecutor
from dataset import ODDataset
from collections import defaultdict
from model import DoublyConstrainedGravityModel
os.path.dirname(os.path.abspath(__file__))
from evaluation.fixed_eval_utils import apply_merge_events

warnings.filterwarnings('ignore')


# ===== 병렬 평가 worker (ThreadPoolExecutor용 — GIL 해제되는 numpy/LGBM은 스레드로 진짜 병렬) =====
def _eval_one_sample(args):
    """(model, base_data, ...) 받아 단일 샘플 평가 → dict 반환"""
    model, base_data, useLog, year_label, city_name, task, split_name, mask_indices, merge_events = args
    
    try:
        sample = apply_merge_events(base_data, mask_indices, merge_events)
        
        # gravity 모델은 항상 raw_static(18 피치)으로 학습
        O_pred, D_pred = model.predict_O_D(sample['X_static_raw'].float().numpy(), useLog)
        
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
                'split': split_name}
    except Exception as e:
        print(f"  [WARN] 샘플 실패 ({city_name} task={task}): {e}")
        return None

def evaluate_and_report(base_data, val_meta, useRaw, model, useLog, year_label, split_name, n_workers=4):
    """ThreadPoolExecutor로 샘플병 병렬 평가"""
    
    job_args = []
    for task in [0, 1, 2, 3, 4]:
        for city_name, val_meta_task_list in val_meta.items():
            for meta in val_meta_task_list[task]:
                job_args.append((
                    model, base_data, useLog,
                    year_label, city_name, task, split_name,
                    meta['mask_indices'], meta['merge_events']
                ))
    
    total = len(job_args)
    print(f"  [{year_label}/{split_name}] {total}개 샘플 병렬 평가 중 (threads={n_workers})...")
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(_eval_one_sample, job_args))
    
    records = [r for r in results if r is not None]
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
    args = parser.parse_args()
    
    year_labels = ['2019', '2023']
    datasets = []
    X_static_train_list = []
    X_o_train_list, X_d_train_list = [], []
    
    for year in year_labels:
        dataset = ODDataset(year=year, imputation=args.imputation, use_raw_static=args.model_type in ('trip_rate', 'cross_class', 'linear_regression'))
        X_static_train_list.append(dataset.X_static_train)
        X_o_train_list.append(dataset.y_o[dataset.train_mask])
        X_d_train_list.append(dataset.y_d[dataset.train_mask])
        datasets.append(dataset)
    
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
            records = evaluate_and_report(base_data, meta_dict, useRaw, model, useLog, year, split_name)
            all_records.extend(records)
            
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

# CPC 점수 함수
def cpc_score(y_t, y_p):
    num = 2 * np.sum(np.minimum(y_t, y_p))
    den = np.sum(y_t) + np.sum(y_p)
    return num / den if den > 0 else 0.0

if __name__ == "__main__":
    main()
