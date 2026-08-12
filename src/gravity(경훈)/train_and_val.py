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
    start_time = time.time()
    progress_every = 30
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
    parser.add_argument('--log_dir', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs'),
                        help='콘솔 출력 로그 저장 폴더')
    parser.add_argument('--train_years', nargs='+', default=['2019', '2023'],
                        choices=['2019', '2023'])
    parser.add_argument('--eval_years', nargs='+', default=None,
                        choices=['2019', '2023'])
    parser.add_argument('--eval_splits', nargs='+', default=['val', 'test'],
                        choices=['val', 'test'])
    args = parser.parse_args()
    if args.eval_years is None:
        args.eval_years = list(args.train_years)
    log_file, log_path = setup_logging(args)
    
    train_year_labels = list(args.train_years)
    eval_year_labels = list(args.eval_years)
    datasets = []
    imputation_values_by_year = {}
    X_static_train_list = []
    X_o_train_list, X_d_train_list = [], []
    
    for year in train_year_labels:
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
    print(f"Start Model Training on years: {', '.join(train_year_labels)}")
    model.fit_O_D(X_static_train, X_o_train, X_d_train, useLog)
    
    # 프로젝트 루트: src/gravity(경훈)/ -> src/ -> 루트
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    fixed_eval_dir = os.path.join(base_dir, "dataset", "fixed_eval")
    all_records = []
    
    for year in eval_year_labels:
        base_data_path = os.path.join(fixed_eval_dir, f"base_data_{year}.pt")
        if not os.path.exists(base_data_path):
            print(f"[SKIP] base_data_{year}.pt not found")
            continue
        base_data = torch.load(base_data_path, weights_only=False)
        
        for split_name in args.eval_splits:
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
