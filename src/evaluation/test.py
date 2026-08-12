import os
import argparse
import torch
import numpy as np
import pandas as pd
import sys

# Define model paths dynamically based on model_type
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from evaluation.evaluation_pipeline import run_evaluation_pipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default=None, help='Path to checkpoint (optional, defaults to latest if not provided and model_type != gravity)')
    parser.add_argument('--model_type', type=str, default='mae', choices=['mae', 'mae-old', 'mae-new', 'twostage', 'gravity'])
    parser.add_argument('--loss_type', type=str, default='hybrid_od', help='Loss function used')
    parser.add_argument('--gravity_model_type', type=str, default='lgbm', help='Gravity generation model type')
    parser.add_argument('--gravity_imputation', type=str, default='zero', help='Gravity imputation method (zero, mean)')
    args = parser.parse_args()
    
    if args.model_type != 'gravity' and args.ckpt_path is None:
        best_model_dir = os.path.join(os.path.dirname(__file__), '../../best_model')
        if not os.path.exists(best_model_dir):
            print(f"Error: {best_model_dir} 경로가 존재하지 않습니다. --ckpt_path를 입력해주세요.")
            sys.exit(1)
        
        ckpt_files = [f for f in os.listdir(best_model_dir) if f.endswith('.pt') or f.endswith('.pth')]
        ckpt_files.sort()
        if not ckpt_files:
            print(f"Error: {best_model_dir}에 가중치(.pt, .pth) 파일이 없습니다.")
            sys.exit(1)
            
        print("\n=== best_model 가중치 목록 ===")
        for i, f in enumerate(ckpt_files):
            print(f"[{i}] {f}")
        print("=============================")
        
        try:
            choice = input(f"사용할 가중치 번호를 입력하세요 (0 ~ {len(ckpt_files)-1}): ")
            choice_idx = int(choice.strip())
            if choice_idx < 0 or choice_idx >= len(ckpt_files):
                raise ValueError
            args.ckpt_path = os.path.join(best_model_dir, ckpt_files[choice_idx])
            print(f"선택된 가중치: {args.ckpt_path}\n")
        except Exception:
            print("올바른 번호를 입력하지 않아 종료합니다.")
            sys.exit(1)
            
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    test_data_path = os.path.join(os.path.dirname(__file__), '../../dataset/fixed_eval/fixed_test_dataset.pt')
    if not os.path.exists(test_data_path):
        print("Test dataset not found. Please run make_val_test_dataset.py first.")
        sys.exit(1)
        
    print(f"Loading {test_data_path}...")
    test_data = torch.load(test_data_path, weights_only=False)
    
    # Extract dimensions from dataset
    sample = list(list(test_data.values())[0].values())[0][0]
    N = sample['X_static'].shape[0]
    num_features = sample['X_static'].shape[1]
    
    model = None
    if args.model_type == 'mae':
        from mae.models import SpatialODMAE
        model = SpatialODMAE(num_nodes=N, num_features=num_features, d_model=128,
                             num_layers=4, nhead=8, loss_type=args.loss_type)
        model.load_state_dict(torch.load(args.ckpt_path, map_location=device))
        model.to(device)
    elif args.model_type == 'mae-old':
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mae-old'))
        from models import SpatialODMAE as SpatialODMAE_old
        model = SpatialODMAE_old(num_nodes=N, num_features=num_features, use_self_loop_predictor=True, loss_type=args.loss_type)
        model.load_state_dict(torch.load(args.ckpt_path, map_location=device))
        model.to(device)
    elif args.model_type == 'mae-new':
        from dataset import CONT_COLS, PROP_MULTI_COLS, PROP_SINGLE_COLS, ZERO_COLS
        from mae_new.models import SpatialODMAE as SpatialODMAE_new
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mae-old'))
        from dataset import ODDataset
        
        dummy_ds = ODDataset(mode='val')
        f_cols = dummy_ds.feature_cols
        
        idx_cont = [f_cols.index(c) for c in CONT_COLS if c in f_cols]
        idx_prop_multi = [f_cols.index(c) for c in PROP_MULTI_COLS if c in f_cols]
        idx_prop_single = [f_cols.index(c) for c in PROP_SINGLE_COLS if c in f_cols]
        idx_zero = [f_cols.index(c) for c in ZERO_COLS if c in f_cols]
        mask_cont_indices = [idx_cont.index(f_cols.index(c)) for c in ['worker_count', 'business_count', 'worker_density', 'business_density'] if c in f_cols and f_cols.index(c) in idx_cont]

        model = SpatialODMAE_new(
            num_static_cont=len(idx_cont),
            num_static_prop_multi=len(idx_prop_multi),
            num_static_prop_single=len(idx_prop_single),
            num_static_zero=len(idx_zero),
            cont_mask_indices=mask_cont_indices,
            use_self_loop_predictor=True
        )
        model.load_state_dict(torch.load(args.ckpt_path, map_location=device))
        model.to(device)
        model.split_indices = {'cont': idx_cont, 'prop_multi': idx_prop_multi, 'prop_single': idx_prop_single, 'zero': idx_zero}
    elif args.model_type == 'twostage':
        from twostage.model import Stage2Model, Stage1Model_LGBM
        import joblib
        
        # Load Stage 2
        model = Stage2Model(num_features=num_features, use_od=False, predict_only_masked=True)
        model.load_state_dict(torch.load(args.ckpt_path, map_location=device))
        model.to(device)
        
        # Load Stage 1
        model.stage1_model = Stage1Model_LGBM(use_4_lgbm=True)
        try:
            model.stage1_model.normal_1 = joblib.load(os.path.join(os.path.dirname(args.ckpt_path), 'lgbm_normal_1.pkl'))
            model.stage1_model.normal_2 = joblib.load(os.path.join(os.path.dirname(args.ckpt_path), 'lgbm_normal_2.pkl'))
            model.stage1_model.masked_1 = joblib.load(os.path.join(os.path.dirname(args.ckpt_path), 'lgbm_masked_1.pkl'))
            model.stage1_model.masked_2 = joblib.load(os.path.join(os.path.dirname(args.ckpt_path), 'lgbm_masked_2.pkl'))
            print("Loaded Stage 1 LGBM models.")
        except Exception as e:
            print(f"Warning: Failed to load Stage 1 LGBM models: {e}. Ensure lgbm_normal_1.pkl etc are in the same dir as the checkpoint.")
            print("Continuing, but predictions may fail if Stage 1 is needed.")
    elif args.model_type == 'gravity':
        model = None
        
    criterion = None
    if args.loss_type == 'hybrid_od':
        from mae.loss import HybridWeightedMSELoss
        criterion = HybridWeightedMSELoss()
        
    print(f"Running evaluation for {args.model_type}...")
    kwargs = {}
    if args.model_type == 'gravity':
        kwargs['gravity_model_type'] = args.gravity_model_type
        kwargs['gravity_imputation'] = args.gravity_imputation
    results = run_evaluation_pipeline(model, test_data, device, model_type=args.model_type, criterion=criterion, **kwargs)
    
    # Organize results by City and Task
    rows = []
    for city, tasks in results.items():
        for task_id, metrics in tasks.items():
            rows.append({
                'City': city,
                'Task': f"Task {task_id}",
                'RMSE': metrics['rmse'],
                'CPC': metrics['cpc'],
                '%RMSE': metrics['prmse']
            })
    df = pd.DataFrame(rows)
    
    # 1. By Task (General) - optional, for debugging
    print("\n=== Detailed Results (By City & Task) ===")
    print(df.pivot(index='City', columns='Task', values='RMSE').round(2))
    
    # 2. Organize by the 3 User Categories:
    # Random: 랜덤1, 랜덤2, 랜덤3
    # New City: 다산, 미사, 배곧, 감일동
    # Comprehensive: All together
    
    random_cities = ['랜덤1', '랜덤2', '랜덤3']
    new_cities = ['다산', '미사', '배곧', '감일동']
    
    def get_stats(df_sub):
        if len(df_sub) == 0:
            return "N/A", "N/A", "N/A"
        rmse_mean, rmse_std = df_sub['RMSE'].mean(), df_sub['RMSE'].std()
        cpc_mean, cpc_std = df_sub['CPC'].mean(), df_sub['CPC'].std()
        prmse_mean, prmse_std = df_sub['%RMSE'].mean(), df_sub['%RMSE'].std()
        return f"{rmse_mean:.2f} ± {rmse_std:.2f}", f"{cpc_mean:.4f} ± {cpc_std:.4f}", f"{prmse_mean:.4f} ± {prmse_std:.4f}"
        
    summary_rows = []
    
    for task_id in [1, 2, 3, 4]:
        task_df = df[df['Task'] == f"Task {task_id}"]
        
        rand_df = task_df[task_df['City'].isin(random_cities)]
        new_df = task_df[task_df['City'].isin(new_cities)]
        comp_df = task_df
        
        r_rmse, r_cpc, r_prmse = get_stats(rand_df)
        n_rmse, n_cpc, n_prmse = get_stats(new_df)
        c_rmse, c_cpc, c_prmse = get_stats(comp_df)
        
        summary_rows.append({
            'Task': f"Task {task_id}",
            'Random RMSE': r_rmse,
            'Random CPC': r_cpc,
            'Random %RMSE': r_prmse,
            'New City RMSE': n_rmse,
            'New City CPC': n_cpc,
            'New City %RMSE': n_prmse,
            'Comprehensive RMSE': c_rmse,
            'Comprehensive CPC': c_cpc,
            'Comprehensive %RMSE': c_prmse
        })
        
    summary_df = pd.DataFrame(summary_rows)
    print("\n=== Final Test Report (Mean ± Std) ===")
    print(summary_df.to_string(index=False))

if __name__ == '__main__':
    main()
