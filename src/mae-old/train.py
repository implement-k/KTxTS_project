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
from loss import WeightedMSELoss, HybridWeightedMSELoss, HuberLoss, HybridWeibullODLoss, BalancedMSELoss, HybridWeightedODLoss
from validation import cpc_score
try:
    import wandb
except ImportError:
    wandb = None
from validation import validate_mae
import lightgbm as lgb
import numpy as np

def str2bool(v):
    return str(v).lower() in ("yes", "true", "t", "1")

def main():
    print("v10-7")
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=TRAIN_CONFIG['epochs'])
    parser.add_argument('--batch_size', type=int, default=TRAIN_CONFIG['batch_size'])
    parser.add_argument('--loss_type', type=str, default='weibull', choices=['weighted_mse', 'hybrid', 'huber', 'weibull', 'bmse', 'hybrid_od']) 
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
                         loss_type=args.loss_type).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    # LR 스케줄과 커리큘럼(alpha/mask) 스케줄 분리
    # 전체 epoch의 80% 시점(예: 70 epoch 중 56 epoch)에서 LR 스케줄을 조기 종료하고,
    # 남은 epoch 동안은 최종 낮은 학습률(1e-5)을 유지하도록 설정
    lr_epochs = int(args.epochs * 0.8)
    lr_total_steps = lr_epochs * len(train_loader)
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=5e-4, 
        total_steps=lr_total_steps,
        pct_start=0.3, 
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=2.0  # max_lr(5e-4) / div_factor(25) = initial_lr(2e-5) -> initial_lr / 2.0 = final_lr(1e-5)
    )
    
    if args.loss_type == 'hybrid': criterion = HybridWeightedMSELoss().to(device)
    elif args.loss_type == 'hybrid_od': criterion = HybridWeightedODLoss().to(device)
    elif args.loss_type == 'bmse': criterion = BalancedMSELoss().to(device)
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
        # Use square root to make curriculum grow much faster in the early epochs
        mask_progress = progress ** 0.5 
        current_mask_size = int(min_mask + (max_mask - min_mask) * mask_progress)
        dataset.max_mask_size = current_mask_size
        current_alpha = max(1.0, 10.0 * progress)

        model.train()
        train_loss = 0.0
        train_y_real_list = []
        train_p_real_list = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Mask:{int(dataset.mask_ratio*100)} α:{current_alpha:.1f}]")
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
                loss, loss_shape, loss_scale, loss_diag, loss_offdiag = criterion(
                    pred_scale, pred_raw, y_od, mask_2d, diag_mask, active_node_mask=active_node_mask, return_components=True, mask_1d=mask)
            elif args.loss_type == 'hybrid_od':
                loss, loss_shape, loss_scale, loss_diag, loss_offdiag = criterion(
                    pred_scale, pred_raw, y_od, current_alpha, mask_2d, diag_mask, active_node_mask=active_node_mask, return_components=True, mask_1d=mask)
            elif args.loss_type == 'huber':
                loss = criterion(pred_raw, y_od, mask=mask_2d)
            else:
                loss = criterion(pred_raw, y_od, current_alpha, mask=mask_2d)
                
            if args.loss_type in ['weibull', 'hybrid_od']:
                pred_raw.retain_grad()
                loss.backward()
                
                if pred_raw.grad is not None:
                    valid_mask_3d = mask.unsqueeze(2).expand_as(pred_raw.grad)
                    valid_grads = pred_raw.grad[valid_mask_3d]
                    pred_raw_grad = valid_grads.abs().mean().item() if valid_grads.numel() > 0 else 0.0
                else:
                    pred_raw_grad = 0.0
                    
                pred_raw_std = pred_raw.std().item()
            else:
                loss.backward()
                
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # 지정된 lr_epochs까지만 스케줄러를 진행하고, 이후엔 최종 LR(1e-5)로 고정
            if epoch < lr_epochs:
                scheduler.step()

            train_loss += loss.item()
            
            with torch.no_grad():
                active_m2d = active_node_mask.unsqueeze(1) & active_node_mask.unsqueeze(2)
                valid_cells = (mask_2d & active_m2d).cpu().numpy()
                y_real_batch = torch.expm1(y_od).cpu().numpy()[valid_cells]
                
                if args.loss_type in ['weibull', 'hybrid_od']:
                    p_real_batch = np.maximum(torch.expm1(pred_scale).cpu().numpy()[valid_cells], 0)
                else:
                    p_real_batch = np.maximum(torch.expm1(pred_raw).cpu().numpy()[valid_cells], 0)
                    
                train_y_real_list.append(y_real_batch)
                train_p_real_list.append(p_real_batch)
            
            if args.loss_type in ['weibull', 'hybrid_od']:
                temp_val = model.temperature.item() if hasattr(model, 'temperature') else 0.0
                pbar.set_postfix({
                    'loss': f"{loss.item():.3f}", 
                    'shape': f"{loss_shape.item():.3f}",
                    'grad_s': f"{pred_raw_grad:.1e}",
                    'std': f"{pred_raw_std:.2f}",
                    'T': f"{temp_val:.2f}"
                })
            else:
                pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{scheduler.get_last_lr()[0]:.1e}"})

        avg_train_loss = train_loss / len(train_loader)
        
        train_y_real_all = np.concatenate(train_y_real_list)
        train_p_real_all = np.concatenate(train_p_real_list)
        if len(train_y_real_all) > 0:
            train_rmse = np.sqrt(np.mean((train_y_real_all - train_p_real_all) ** 2))
            train_cpc = cpc_score(train_y_real_all, train_p_real_all)
        else:
            train_rmse, train_cpc = 0.0, 0.0
            
        print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f} | Train RMSE: {train_rmse:.2f} | Train CPC: {train_cpc:.4f}")

        # Validation (2 epoch 마다)
        if epoch % 2 == 1 or epoch == args.epochs - 1:
            if 'val_data' not in locals():
                val_data_path = os.path.join(os.path.dirname(__file__), '../../dataset/fixed_eval/fixed_val_dataset.pt')
                if not os.path.exists(val_data_path):
                    print(f"Warning: {val_data_path} not found. Skipping validation.")
                    continue
                val_data = torch.load(val_data_path, map_location='cpu')
            
            from KTDB.src.evaluation.evaluation_pipeline import run_evaluation_pipeline
            print(f"\n--- Validation Epoch {epoch+1} ---")
            
            val_results = run_evaluation_pipeline(
                model=model,
                data_dict=val_data,
                device=device,
                model_type='mae',
                criterion=criterion
            )
            
            total_rmse, total_cpc, valid_tasks = 0, 0, 0
            
            for city, tasks in val_results.items():
                print(f"  [{city}] Task 1(Mask): Loss:{tasks[1]['loss']:.4f} RMSE:{tasks[1]['rmse']:.2f} CPC:{tasks[1]['cpc']:.4f}")
                print(f"  [{city}] Task 2(Merge Full): Loss:{tasks[2]['loss']:.4f} RMSE:{tasks[2]['rmse']:.2f} CPC:{tasks[2]['cpc']:.4f}")
                print(f"  [{city}] Task 3(Merge Partial Context): Loss:{tasks[3]['loss']:.4f} RMSE:{tasks[3]['rmse']:.2f} CPC:{tasks[3]['cpc']:.4f}")
                print(f"  [{city}] Task 4(Mask With Known): Loss:{tasks[4]['loss']:.4f} RMSE:{tasks[4]['rmse']:.2f} CPC:{tasks[4]['cpc']:.4f}")
                
                total_rmse += (tasks[1]['rmse'] + tasks[2]['rmse'] + tasks[3]['rmse'] + tasks[4]['rmse'])
                total_cpc += (tasks[1]['cpc'] + tasks[2]['cpc'] + tasks[3]['cpc'] + tasks[4]['cpc'])
                valid_tasks += 4
            
            # Weighted average for best checkpointing (equal weight across all tasks and cities)
            rmse = total_rmse / valid_tasks
            cpc = total_cpc / valid_tasks
            
            if args.use_wandb and wandb.run is not None:
                log_dict = {
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "train_rmse": train_rmse,
                    "train_cpc": train_cpc,
                    "val_rmse_avg": rmse,
                    "val_cpc_avg": cpc,
                    "lr": scheduler.get_last_lr()[0],
                    "mask_ratio": dataset.mask_ratio,
                    "alpha": current_alpha
                }
                for city, tasks in val_results.items():
                    log_dict[f"val_rmse_T1_{city}"] = tasks[1]['rmse']
                    log_dict[f"val_cpc_T1_{city}"] = tasks[1]['cpc']
                    log_dict[f"val_rmse_T2_{city}"] = tasks[2]['rmse']
                    log_dict[f"val_cpc_T2_{city}"] = tasks[2]['cpc']
                    log_dict[f"val_rmse_T3_{city}"] = tasks[3]['rmse']
                    log_dict[f"val_cpc_T3_{city}"] = tasks[3]['cpc']
                    log_dict[f"val_rmse_T4_{city}"] = tasks[4]['rmse']
                    log_dict[f"val_cpc_T4_{city}"] = tasks[4]['cpc']
                wandb.log(log_dict)

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
