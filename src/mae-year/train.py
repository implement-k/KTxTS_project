import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRAIN_CONFIG

import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import ODDataset
from models import ODMAE
from tqdm import tqdm
from loss import WeightedMSELoss, HybridWeightedMSELoss, HuberLoss
import wandb
from validation import evaluate_and_report, summarize_results
from evaluation.fixed_eval_utils import make_base_data
import lightgbm as lgb
import numpy as np

if hasattr(torch.backends, "mha"):
      torch.backends.mha.set_fastpath_enabled(False)

def str2bool(v):
    return str(v).lower() in ("yes", "true", "t", "1")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=TRAIN_CONFIG['epochs'])
    parser.add_argument('--batch_size', type=int, default=TRAIN_CONFIG['batch_size'])
    parser.add_argument('--loss_type', type=str, default='weighted_mse', choices=['weighted_mse', 'hybrid', 'huber']) # v1, v2, v3: weighted_mse
    parser.add_argument('--od_embed_layers', type=int, default=3)                   # v1: 1, v2: 2, v3, v4: 3
    parser.add_argument('--use_friction', type=str2bool, default=True)              # v1, v2, v3: False, v4: True
    parser.add_argument('--use_self_loop_predictor', type=str2bool, default=True)   # v1: False, v2, v3, v4: True
    parser.add_argument('--lambda_diag', type=float, default=1.0)                   # v6: 50(수치상으로는 130이 맞긴함)
    parser.add_argument('--use_lgbm_self_loop', type=str2bool, default=False)       # v7: True
    parser.add_argument('--use_mask_channel', type=str2bool, default=False)         # v6~: True  
    parser.add_argument('--use_wandb', type=str2bool, default=False)
    parser.add_argument('--wandb_id', type=str, default=None, help="기존 wandb run id (이어서 학습 시)")
    parser.add_argument('--use_stratfied_masking', type=str2bool, default=True)
    parser.add_argument('--use_merge_train', type=str2bool, default=True, help="행정동 병합 학습 여부")
    parser.add_argument('--use_od_log_transform', type=str2bool, default=True, help="X_OD에 대해 log1p 적용 여부")
    parser.add_argument('--use_static_normalize', type=str2bool, default=True, help="X_static에 대해 z-score normalization 적용 여부")
    parser.add_argument('--use_dist_log_transform', type=str2bool, default=True, help="distance matrix에 log1p 적용 여부")
    parser.add_argument('--year', type=str, default='2023', choices=['2019', '2023'], help="학습할 연도")
    return parser.parse_args()

def main():
    # === arg parsing ===
    args = parse_args()
    # v5 train.py --epochs 70 --batch_size 32 --od_embed_layers 2 use_friction False --use_self_loop_predictor False  --lambda_diag -1.0 --use_lgbm_self_loop False --use_mask_channel True --use_wandb True
    
    if args.use_wandb: 
        if args.wandb_id:
            wandb.init(project="MAE", id=args.wandb_id, resume="allow", config=vars(args))
        else:
            wandb.init(project="MAE", config=vars(args))
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps'  if torch.backends.mps.is_available() else 'cpu')
    
    print("선택된 argument:")
    for arg in vars(args): print(f"  {arg}: {getattr(args, arg)}")
    year_labels = [args.year]
    dataset_dict, train_loaders = {}, {}
    fixed_eval_dir = os.path.join(os.path.dirname(__file__), '../../dataset/fixed_eval')
    base_data_dict = {}
    val_meta_dict = {}

    # === dataset 로드 ===
    for year in year_labels:
        # === train dataset 로드 ===
        dataset_dict[year] = ODDataset(year=year, 
                                       use_stratfied_masking=args.use_stratfied_masking, 
                                       use_merge_train=args.use_merge_train, 
                                       use_od_log_transform=args.use_od_log_transform, 
                                       use_static_normalize=args.use_static_normalize,
                                       use_dist_log_transform=args.use_dist_log_transform)
    
        train_loaders[year] = DataLoader(dataset_dict[year], batch_size=args.batch_size, shuffle=True)

        # === validation dataset 로드 ===
        meta_data_path = os.path.join(fixed_eval_dir, f"fixed_val_meta_{year}.pt")
        
        if not os.path.exists(meta_data_path):
            raise FileNotFoundError(f"E: fixed_val_meta_{year}.pt 파일이 없습니다: {meta_data_path}")

        # 메모리 절약을 위해 base_data.pt를 디스크에서 로드하지 않고 현재 로드된 dataset에서 직접 생성
        base_data = make_base_data(dataset_dict[year])
        base_data_dict[year] = base_data
        
        meta_data = torch.load(meta_data_path, weights_only=False)
        val_meta_dict[year] = meta_data
    
    F = dataset_dict[args.year].X_static.shape[1]

    model = ODMAE(num_features=F, 
                od_embed_layers=args.od_embed_layers,
                use_distance_friction=args.use_friction,
                use_self_loop_predictor=args.use_self_loop_predictor,
                use_mask_channel=args.use_mask_channel).to(device)

    # wandb_id가 주어지고 마지막 체크포인트가 존재하면 가중치 로드
    current_dir = os.path.dirname(os.path.abspath(__file__))
    last_model_path = os.path.join(current_dir, 'last_model_mae.pth')
    if args.wandb_id and os.path.exists(last_model_path):
        print(f"\n[Resume] 기존 체크포인트를 불러옵니다: {last_model_path}")
        model.load_state_dict(torch.load(last_model_path, map_location=device))
        
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    # 두 dataloader 길이는 같음 (dataset __len__이 1000으로 고정)
    total_steps = args.epochs * sum(len(loader) for loader in train_loaders.values())
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=5e-4, 
        total_steps=total_steps,
        pct_start=0.3, 
        anneal_strategy='cos'
    )
    
    if args.loss_type == 'hybrid': criterion = HybridWeightedMSELoss().to(device)
    elif args.loss_type == 'huber': criterion = HuberLoss().to(device)
    else: criterion = WeightedMSELoss().to(device)

    best_val_rmse = float('inf')
    best_cpc = 0.0
    best_model_path = f'best_model_mae_{args.year}.pth'

    min_mask = TRAIN_CONFIG['min_mask_size']
    max_mask = TRAIN_CONFIG['max_mask_size']
    
    for epoch in range(args.epochs):
        progress = epoch / max(1, args.epochs - 1)
        current_mask_size = int(min_mask + (max_mask - min_mask) * progress)
        for ds in dataset_dict.values():
            ds.max_mask_size = current_mask_size
        current_alpha = min(10.0, 1.0 + 9.0 * progress)

        model.train()
        train_loss = 0

        def batch_generator():
            # Batch iteration
            for batches in zip(*train_loaders.values()):
                for batch in batches:
                    yield batch
                    
        total_batches = sum(len(loader) for loader in train_loaders.values())
        pbar = tqdm(batch_generator(), total=total_batches, desc=f"Epoch {epoch+1}/{args.epochs} [Mask:{current_mask_size} α:{current_alpha:.1f}]")
        for batch in pbar:
            x_static = batch['X_static'].to(device)
            x_dist = batch['X_dist'].to(device)
            a_spatial = batch['A_spatial'].to(device)
            mask = batch['mask'].to(device)
            x_od_masked = batch['X_OD_masked'].to(device)
            y_od = batch['y_OD'].to(device)
            active_node_mask = batch['active_node_mask'].to(device)

            optimizer.zero_grad()
            pred = model(x_static, x_od_masked, x_dist, a_spatial, mask, active_node_mask)
            
            # pred shape에서 동의 개수 유추
            N_nodes = pred.shape[1]
            diag_mask = torch.eye(N_nodes, device=device, dtype=torch.bool).unsqueeze(0).expand(pred.shape[0], -1, -1)
            active_mask_2d = active_node_mask.unsqueeze(1) & active_node_mask.unsqueeze(2)
            mask_2d = (mask.unsqueeze(1) | mask.unsqueeze(2)) & active_mask_2d

            if args.lambda_diag < 0:
                # 원래 방식: 대각/비대각 구분 없이 한 번에 평균
                loss = criterion(pred, y_od, current_alpha, mask_2d)
            else:
                valid_diag_mask = diag_mask & mask_2d
                valid_offdiag_mask = (~diag_mask) & mask_2d
                
                loss_diag = criterion(pred, y_od, current_alpha, valid_diag_mask) if valid_diag_mask.any() else 0.0
                loss_offdiag = criterion(pred, y_od, current_alpha, valid_offdiag_mask) if valid_offdiag_mask.any() else 0.0
                
                loss = loss_offdiag + (args.lambda_diag * loss_diag)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{scheduler.get_last_lr()[0]:.1e}"})

        avg_train_loss = train_loss / total_batches
        print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")

        # Validation (2 epoch 마다)
        if (epoch % 2 == 1 or epoch == args.epochs - 1):
            all_records = []
            
            for year in year_labels:
                if year in base_data_dict and year in val_meta_dict:
                    records = evaluate_and_report(
                        base_data=base_data_dict[year], 
                        val_meta=val_meta_dict[year], 
                        model=model, 
                        year_label=year, 
                        split_name='val', 
                        n_workers=1, 
                        device=device
                    )
                    all_records.extend(records)
            
            if all_records:
                print("\n=== Validation Results ===")
                summarize_results(all_records, ['year', 'task'], "연도별 task 요약")
                summarize_results(all_records, ['year'], "연도별 총 요약")
                
                rmse = np.mean([r['rmse'] for r in all_records])
                cpc = np.mean([r['cpc'] for r in all_records])
                prmse = np.mean([r['prmse'] for r in all_records])
                
                print(f"  ➜ [Val] RMSE: {rmse:.2f} | CPC: {cpc:.4f} | PRMSE: {prmse:.4f}")
            else:
                rmse = float('inf')
                cpc = 0.0
                prmse = 0.0

            if args.use_wandb:
                log_dict = {
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "val_rmse": rmse,
                    "val_cpc": cpc,
                    "val_prmse": prmse
                }
                
                # Task 별 메트릭 계산 및 추가
                if all_records:
                    for t in range(5):
                        t_records = [r for r in all_records if r['task'] == t]
                        if t_records:
                            log_dict[f"val_rmse_task{t}"] = np.mean([r['rmse'] for r in t_records])
                            log_dict[f"val_cpc_task{t}"] = np.mean([r['cpc'] for r in t_records])
                            log_dict[f"val_prmse_task{t}"] = np.mean([r['prmse'] for r in t_records])
                            
                wandb.log(log_dict)
            if rmse < best_val_rmse:
                best_val_rmse = rmse
                current_dir = os.path.dirname(os.path.abspath(__file__))
                torch.save(model.state_dict(),
                           os.path.join(current_dir, best_model_path))
                print(f"  ➜ [Checkpoint] Best RMSE saved! (RMSE:{rmse:.2f} CPC:{cpc:.4f} PRMSE:{prmse:.4f})")

            if cpc > best_cpc:
                best_cpc = cpc
                current_dir = os.path.dirname(os.path.abspath(__file__))
                torch.save(model.state_dict(),
                           os.path.join(current_dir, f'best_model_mae_cpc_{args.year}.pth'))
                print(f"  ➜ [Checkpoint] Best CPC saved! (RMSE:{rmse:.2f} CPC:{cpc:.4f} PRMSE:{prmse:.4f})")

            model.train()

        # 에포크 종료 시마다 last checkpoint 저장 및 wandb 업로드
        current_dir = os.path.dirname(os.path.abspath(__file__))
        last_model_path = os.path.join(current_dir, f'last_model_mae_{args.year}.pth')
        torch.save(model.state_dict(), last_model_path)
        if args.use_wandb:
            wandb.save(last_model_path, base_path=current_dir)

    print(f"\nTraining Complete. Best RMSE: {best_val_rmse:.2f} | Best CPC: {best_cpc:.4f} | Best PRMSE: {prmse:.4f}")
    
    if args.use_lgbm_self_loop:
        print(f"\nLGBM 모델 학습 시작 ({args.year} 데이터 기준)")
        
        train_idx = dataset_dict[args.year].train_indices
        
        X_train_lgb = dataset_dict[args.year].X_static[train_idx]
        y_train_lgb = np.diag(dataset_dict[args.year].X_OD)[train_idx]
        
        lgbm_model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
        lgbm_model.fit(X_train_lgb, y_train_lgb)
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        lgbm_path = os.path.join(current_dir, '../best_model/best_lgbm_self_loop.txt')
        os.makedirs(os.path.dirname(lgbm_path), exist_ok=True)
        lgbm_model.booster_.save_model(lgbm_path)
        print(f"LGBM 모델 저장: {lgbm_path}")

    if args.use_wandb: wandb.finish()

if __name__ == '__main__':
    main()
