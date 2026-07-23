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
from models import SpatialODMAE
from tqdm import tqdm
from loss import WeightedMSELoss, HybridWeightedMSELoss, HuberLoss, HybridWeibullODLoss
import wandb
from validation_mae import validate_mae
import lightgbm as lgb
import numpy as np

def str2bool(v):
    return str(v).lower() in ("yes", "true", "t", "1")

def main():
    print("v10")
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=TRAIN_CONFIG['epochs'])
    parser.add_argument('--batch_size', type=int, default=TRAIN_CONFIG['batch_size'])
    parser.add_argument('--loss_type', type=str, default='weibull', choices=['weighted_mse', 'hybrid', 'huber', 'weibull']) 
    parser.add_argument('--use_self_loop_predictor', type=str2bool, default=True)   
    parser.add_argument('--use_lgbm_self_loop', type=str2bool, default=False)   
    parser.add_argument('--use_wandb', type=str2bool, default=False)
    parser.add_argument('--use_mask_channel', type=str2bool, default=True)
    parser.add_argument('--resume_checkpoint', type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument('--wandb_run_id', type=str, default=None, help="WandB run ID to resume")
    args = parser.parse_args()
    
    if args.use_wandb: wandb.init(project="SpatialODMAE", config=vars(args), id=args.wandb_run_id, resume="allow")
    
    print("선택된 argument:")
    for arg in vars(args): print(f"  {arg}: {getattr(args, arg)}")

    dataset = ODDataset(mode='train')
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps'  if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    min_mask = TRAIN_CONFIG['min_mask_size']
    max_mask = TRAIN_CONFIG['max_mask_size']

    # Validation 대상은 Val 도시 전체
    val_indices = dataset.val_indices
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = SpatialODMAE(num_nodes=dataset.num_nodes, 
                         num_features=dataset.X_static.shape[1],
                         use_self_loop_predictor=args.use_self_loop_predictor,
                         use_mask_channel=args.use_mask_channel).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    total_steps = args.epochs * len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=5e-4, 
        total_steps=total_steps,
        pct_start=0.3, 
        anneal_strategy='cos'
    )
    
    if args.loss_type == 'hybrid': criterion = HybridWeightedMSELoss().to(device)
    elif args.loss_type == 'huber': criterion = HuberLoss().to(device)
    elif args.loss_type == 'weibull': criterion = HybridWeibullODLoss().to(device)
    else: criterion = WeightedMSELoss().to(device)

    start_epoch = 0
    best_val_rmse = float('inf')
    best_cpc = 0.0
    best_model_path = 'best_model_mae.pth'
    last_checkpoint_path = 'last_checkpoint.pth'

    if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
        print(f"Resuming from checkpoint: {args.resume_checkpoint}")
        checkpoint = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_rmse = checkpoint.get('best_val_rmse', float('inf'))
        best_cpc = checkpoint.get('best_cpc', 0.0)
        print(f"Resumed at epoch {start_epoch}, best RMSE: {best_val_rmse:.2f}, best CPC: {best_cpc:.4f}")

    for epoch in range(start_epoch, args.epochs):
        progress = epoch / max(1, args.epochs - 1)
        current_mask_size = int(min_mask + (max_mask - min_mask) * progress)
        dataset.max_mask_size = current_mask_size
        current_alpha = max(1.0, 10.0 * (1.0 - progress))

        model.train()
        train_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} " f"[Mask:{current_mask_size} α:{current_alpha:.1f}]")
        for batch in pbar:
            x_static = batch['X_static'].to(device)
            x_dist = batch['X_dist'].to(device)
            mask = batch['mask'].to(device)
            x_od_masked = batch['X_OD_masked'].to(device)
            y_od = batch['y_OD'].to(device)
            
            # Fetch active_node_mask if it exists (for backward compatibility with old dataset output)
            if 'active_node_mask' in batch:
                active_node_mask = batch['active_node_mask'].to(device)
            else:
                active_node_mask = torch.ones_like(mask, dtype=torch.bool, device=device)

            optimizer.zero_grad()
            # pred from model was crashing here before because it used `pred.shape` not `pred_scale.shape`
            pred_scale, pred_raw = model(x_static, x_od_masked, x_dist, mask, active_node_mask=active_node_mask)
            
            diag_mask = torch.eye(dataset.num_nodes, device=device, dtype=torch.bool).unsqueeze(0).expand(pred_scale.shape[0], -1, -1)
            mask_2d = mask.unsqueeze(1) | mask.unsqueeze(2)

            if args.loss_type == 'weibull':
                loss = criterion(pred_scale, pred_raw, y_od, mask_2d, diag_mask, active_node_mask=active_node_mask)
            else:
                loss = criterion(pred_raw, y_od, current_alpha, mask_2d)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{scheduler.get_last_lr()[0]:.1e}"})

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")

        # Validation (2 epoch 마다)
        if epoch % 2 == 1 or epoch == args.epochs - 1:
            from validation_mae import validate_mae_merge
            v_loss_a, rmse_a, cpc_a = validate_mae(model, dataset, val_indices, criterion, device)
            v_loss_b, rmse_b, cpc_b = validate_mae_merge(model, dataset, val_indices, criterion, device)
            
            print(f"  ➜ [Val Task A: Masking] Loss: {v_loss_a:.4f} | RMSE: {rmse_a:.2f} | CPC: {cpc_a:.4f}")
            print(f"  ➜ [Val Task B: Merge  ] Loss: {v_loss_b:.4f} | RMSE: {rmse_b:.2f} | CPC: {cpc_b:.4f}")
            
            # Weighted average for best checkpointing (equal weight)
            rmse = (rmse_a + rmse_b) / 2.0
            cpc = (cpc_a + cpc_b) / 2.0
            
            if args.use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "val_loss_A": v_loss_a,
                    "val_rmse_A": rmse_a,
                    "val_cpc_A": cpc_a,
                    "val_loss_B": v_loss_b,
                    "val_rmse_B": rmse_b,
                    "val_cpc_B": cpc_b,
                    "val_rmse_avg": rmse,
                    "val_cpc_avg": cpc
                })

            if rmse < best_val_rmse:
                best_val_rmse = rmse
                current_dir = os.path.dirname(os.path.abspath(__file__))
                torch.save(model.state_dict(),
                           os.path.join(current_dir, best_model_path))
                print(f"  ➜ [Checkpoint] Best RMSE saved! (RMSE:{rmse:.2f} CPC:{cpc:.4f})")

            if cpc > best_cpc:
                best_cpc = cpc
                current_dir = os.path.dirname(os.path.abspath(__file__))
                torch.save(model.state_dict(),
                           os.path.join(current_dir, 'best_model_mae_cpc.pth'))
                print(f"  ➜ [Checkpoint] Best CPC saved! (RMSE:{rmse:.2f} CPC:{cpc:.4f})")

            model.train()
            
        # 매 에포크마다 재개를 위한 상태 저장
        current_dir = os.path.dirname(os.path.abspath(__file__))
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_rmse': best_val_rmse,
            'best_cpc': best_cpc,
        }, os.path.join(current_dir, last_checkpoint_path))

    print(f"\nTraining Complete. Best RMSE: {best_val_rmse:.2f} | Best CPC: {best_cpc:.4f}")
    
    if args.use_lgbm_self_loop:
        print("\nLGBM 모델 학습 시작")
        
        train_idx = dataset.train_indices
        X_train_lgb = dataset.X_static[train_idx]
        y_train_lgb = np.diag(dataset.X_OD)[train_idx]
        
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
