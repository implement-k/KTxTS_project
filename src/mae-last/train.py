import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRAIN_CONFIG

import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from dataset import ODDataset
from models import ODMAE
from tqdm import tqdm
from loss import WeightedMSELoss, HybridWeightedMSELoss, HuberLoss
import wandb
from validation import evaluate_and_report, summarize_results
import lightgbm as lgb
import numpy as np
import random

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

if hasattr(torch.backends, "mha"):
      torch.backends.mha.set_fastpath_enabled(False)

def str2bool(v):
    return str(v).lower() in ("yes", "true", "t", "1")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=TRAIN_CONFIG['epochs'])
    parser.add_argument('--batch_size', type=int, default=TRAIN_CONFIG['batch_size'])
    parser.add_argument('--loss_type', type=str, default='hybrid', choices=['weighted_mse', 'hybrid', 'huber'])           
    parser.add_argument('--use_self_loop_predictor', type=str2bool, default=True)                 
    parser.add_argument('--use_lgbm_self_loop', type=str2bool, default=False)       
    parser.add_argument('--use_wandb', type=str2bool, default=False)
    parser.add_argument('--wandb_id', type=str, default=None, help="기존 wandb run id (이어서 학습 시)")
    parser.add_argument('--use_stratfied_masking', type=str2bool, default=True)
    parser.add_argument('--use_merge_train', type=str2bool, default=True, help="행정동 병합 학습 여부")
    parser.add_argument('--use_transformer', type=str2bool, default=True, help="Transformer 대신 FFN 사용 (Ablation)")
    parser.add_argument('--od_scale_ablation', type=str, default='none', choices=['none', 'zero', 'global'], help="OD scale GCN ablation mode")
    parser.add_argument('--year', type=str, default='2023', choices=['2019', '2023'], help="학습할 연도")
    parser.add_argument('--dist_dropout_max_prob', type=float, default=0.3)
    parser.add_argument('--use_rle', type=str2bool, default=False)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()

def main():
    # === arg parsing ===
    args = parse_args()
    # 최종 train.py --epochs 90 --batch_size 32 --use_self_loop_predictor False --use_lgbm_self_loop False --use_wandb True
    
    if args.use_wandb: 
        if args.wandb_id:
            wandb.init(project="MAE", id=args.wandb_id, resume="allow", config=vars(args))
        else:
            wandb.init(project="MAE", config=vars(args))
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps'  if torch.backends.mps.is_available() else 'cpu')
    
    # === 0. Seed 설정 ===
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    print("선택된 argument:")
    for arg in vars(args): print(f"  {arg}: {getattr(args, arg)}")
    year = args.year
    dataset_dict, train_loaders = {}, {}
    fixed_eval_dir = os.path.join(os.path.dirname(__file__), '../../dataset/fixed_eval')
    base_data_dict = {}
    val_meta_dict = {}

    # === dataset 로드 ===
    # === train dataset 로드 ===
    dataset_dict[year] = ODDataset(year=year, 
                                    use_stratfied_masking=args.use_stratfied_masking, 
                                    use_merge_train=args.use_merge_train)

    train_loaders[year] = DataLoader(dataset_dict[year], batch_size=args.batch_size, shuffle=True, num_workers=4, worker_init_fn=worker_init_fn)

    # === validation dataset 로드 ===
    base_data_path = os.path.join(fixed_eval_dir, f"base_data_{year}.pt")
    meta_data_path = os.path.join(fixed_eval_dir, f"fixed_val_meta_{year}.pt")
    
    if not os.path.exists(meta_data_path):
        raise FileNotFoundError(f"E: fixed_val_meta_{year}.pt 파일이 없습니다: {meta_data_path}")

    # 메모리 절약을 위해 base_data.pt를 디스크에서 로드하지 않고 현재 로드된 dataset에서 직접 생성
    base_data = torch.load(base_data_path, weights_only=False)
    base_data_dict[year] = base_data
    
    meta_data = torch.load(meta_data_path, weights_only=False)
    val_meta_dict[year] = meta_data
    
    F = dataset_dict[args.year].X_static.shape[1]

    model = ODMAE(num_features=F, 
                  use_self_loop_predictor=args.use_self_loop_predictor, 
                  use_transformer=args.use_transformer,
                  od_scale_ablation=args.od_scale_ablation,
                  use_rle=args.use_rle).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
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
    
    start_epoch = 0

    # wandb_id가 주어지고 마지막 체크포인트가 존재하면 전체 상태 로드
    current_dir = os.path.dirname(os.path.abspath(__file__))
    last_model_path = os.path.join(current_dir, f'last_model_mae_{args.year}.pth')
    if args.wandb_id and os.path.exists(last_model_path):
        print(f"\n[Resume] 기존 체크포인트를 불러옵니다: {last_model_path}")
        checkpoint = torch.load(last_model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_rmse = checkpoint.get('best_val_rmse', float('inf'))
            best_cpc = checkpoint.get('best_cpc', 0.0)
            print(f"  ➜ {start_epoch} Epoch부터 이어서 학습을 시작합니다.")
        else:
            print("W: 구버전 체크포인트입니다. 가중치만 불러옵니다 (Optimizer, Scheduler 등은 초기화됩니다).")
            model.load_state_dict(checkpoint)
    
    for epoch in range(start_epoch, args.epochs):
        progress = epoch / max(1, args.epochs - 1)
        
        current_mask_size = int(min_mask + (max_mask - min_mask) * progress)
        for ds in dataset_dict.values():
            ds.max_mask_size = current_mask_size
        current_alpha = min(10.0, 1.0 + 9.0 * progress)
        
        # dist_dropout curriculum
        model.dist_dropout_prob = args.dist_dropout_max_prob * progress
        
        if isinstance(criterion, HybridWeightedMSELoss):
            # real_penalty_weight를 점진적으로 증가 (0.003 -> 0.010)
            current_penalty = 0.003 + (0.007 * progress)
            criterion.real_penalty_weight = current_penalty
            
        model.train()
        train_loss = 0

        def batch_generator():
            # Batch iteration
            for batches in zip(*train_loaders.values()):
                for batch in batches:
                    yield batch
                    
        total_batches = sum(len(loader) for loader in train_loaders.values())
        pbar = tqdm(batch_generator(), total=total_batches, desc=f"Epoch {epoch+1}/{args.epochs} [Mask:{current_mask_size} α:{current_alpha:.1f} DDrop:{model.dist_dropout_prob:.2f} Pen:{getattr(criterion, 'real_penalty_weight', 0):.4f}]")
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
            
            mask_2d = mask.unsqueeze(1) | mask.unsqueeze(2)

            loss = criterion(pred, y_od, current_alpha, mask_2d)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{scheduler.get_last_lr()[0]:.1e}"})

        avg_train_loss = train_loss / total_batches
        print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")

        # Validation (2 epoch 마다)
        if (epoch % 2 == 1 or epoch == args.epochs - 1):
            all_records = []
            
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
                summarize_results(all_records, ['task'], "연도별 task 요약")
                summarize_results(all_records, ['city'], "도시별 task 요약")
                summarize_results(all_records, [], "연도별 총 요약")
                
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

        # 에포크 종료 시마다 last checkpoint 전체 상태 저장 및 wandb 업로드
        current_dir = os.path.dirname(os.path.abspath(__file__))
        last_model_path = os.path.join(current_dir, f'last_model_mae_{args.year}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_rmse': best_val_rmse,
            'best_cpc': best_cpc
        }, last_model_path)
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
