import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRAIN_CONFIG, DATA_DIR

import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import MultiRegionDataset
from mae.models import SpatialODMAE
from tqdm import tqdm
try:import wandb
except ImportError: wandb = None
from loss import WeightedMSELoss, HybridWeightedMSELoss, HuberLoss, OffDiagCPCLoss, CrossEntropyODLoss, HuberODLoss
from validation import validate_mae
import lightgbm as lgb
import numpy as np
import torch.nn.functional as F

######### util ##########
def str2bool(v):
    return str(v).lower() in ("yes", "true", "t", "1")

def parse_args(parser):
    parser.add_argument('--epochs', type=int, default=TRAIN_CONFIG['epochs'])
    parser.add_argument('--batch_size', type=int, default=TRAIN_CONFIG['batch_size'])
    parser.add_argument('--loss_type', type=str, default='huber_od', choices=['weighted_mse', 'hybrid', 'huber', 'offdiag_cpc', 'ce_od', 'huber_od'])
    parser.add_argument('--use_self_loop_predictor', type=str2bool, default=True)  
    parser.add_argument('--lambda_diag', type=float, default=-1.0)                   
    parser.add_argument('--use_lgbm_self_loop', type=str2bool, default=True)       
    parser.add_argument('--use_wandb', type=str2bool, default=False)
    parser.add_argument('--resume', type=str2bool, default=False, help="이전 checkpoint로부터 이어서 시작")
    parser.add_argument('--wandb_run_id', type=str, default=None, help="재개할 wandb run id (예: mae:v9-l)")
    parser.add_argument('--tau', type=float, default=0.5, help="Huber Loss 크기 가중치")
    parser.add_argument('--lambda_prior', type=float, default=0.5, help="Meta-Gravity prior Loss 가중치")
    return parser.parse_args()
    
def add_region() -> list[str]:
    # 서울만 사용 (타 지역 제거 시 학습 안정성 확보)
    regions = ['seoul']
    print(f"학습에 사용할 지역 목록: {regions}")
    return regions


def main():
    print("test v10-physics-prior")
    args = parse_args(argparse.ArgumentParser())
    
    print("선택된 argument:")
    for arg in vars(args): print(f"  {arg}: {getattr(args, arg)}")

    regions = add_region()
    
    dataset = MultiRegionDataset(regions=regions, batch_size=args.batch_size, mode='train')
    val_dataset = MultiRegionDataset(regions=regions, batch_size=args.batch_size, mode='val')
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps'  if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    min_mask = TRAIN_CONFIG['min_mask_size']
    max_mask = TRAIN_CONFIG['max_mask_size']

    train_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
    
    ds_seoul = dataset.datasets['seoul']
    model = SpatialODMAE(
        num_static_cont=ds_seoul.X_cont.shape[1],
        num_static_prop_multi=ds_seoul.X_prop_multi.shape[1],
        num_static_prop_single=ds_seoul.X_prop_single.shape[1],
        num_static_zero=ds_seoul.X_zero.shape[1],
        cont_mask_indices=ds_seoul.mask_cont_indices,
        use_self_loop_predictor=args.use_self_loop_predictor
    ).to(device)
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
    elif args.loss_type == 'offdiag_cpc': criterion = OffDiagCPCLoss(diag_weight=0.1, cpc_weight=0.5).to(device)
    elif args.loss_type == 'ce_od': criterion = CrossEntropyODLoss().to(device)
    elif args.loss_type == 'huber_od': criterion = HuberODLoss(delta=1.0, tau=args.tau).to(device)
    else: criterion = WeightedMSELoss().to(device)
    
    best_val_rmse = float('inf')
    best_cpc = 0.0
    best_model_path = 'best_model_mae.pth'
    start_epoch = 0
    wandb_run_id = None

    if args.resume:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        latest_ckpt_path = os.path.join(current_dir, 'checkpoint_latest.pth')
        if os.path.exists(latest_ckpt_path):
            print(f"Resuming from {latest_ckpt_path}...")
            checkpoint = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_rmse = checkpoint.get('best_val_rmse', float('inf'))
            best_cpc = checkpoint.get('best_cpc', 0.0)
            wandb_run_id = checkpoint.get('wandb_run_id', None)
            print(f"재개 성공. epoch: {start_epoch}. Best RMSE: {best_val_rmse:.2f}, Best CPC: {best_cpc:.4f}")
        else:
            print("체크포인트 없음. 처음부터 시작")

    if args.wandb_run_id:
        wandb_run_id = args.wandb_run_id

    if args.use_wandb:
        wandb.init(
            project="SpatialODMAE",
            config=vars(args),
            resume="allow",
            id=wandb_run_id
        )

    for epoch in range(start_epoch, args.epochs):
        progress = epoch / max(1, args.epochs - 1)
        current_mask_size = int(min_mask + (max_mask - min_mask) * progress)
        dataset.max_mask_size = current_mask_size
        current_alpha = max(1.0, 10.0 * (1.0 - progress))

        model.train()
        train_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} " f"[Mask:{current_mask_size} α:{current_alpha:.1f}]")
        for _, batch in enumerate(pbar):
            x_cont = batch['X_cont'].squeeze(0).to(device)
            x_prop_multi = batch['X_prop_multi'].squeeze(0).to(device)
            x_prop_single = batch['X_prop_single'].squeeze(0).to(device)
            x_zero = batch['X_zero'].squeeze(0).to(device)
            
            x_dist = batch['X_dist'].squeeze(0).to(device)
            x_od_masked = batch['X_OD_masked'].squeeze(0).to(device)
            y_od = batch['y_OD'].squeeze(0).to(device)
            mask = batch['mask'].squeeze(0).to(device)
            loss_mask = batch['loss_mask'].squeeze(0).to(device)
            pop_raw = batch['pop_raw'].squeeze(0).to(device)
            
            optimizer.zero_grad()
            pred_od, prior_od = model(x_cont, x_prop_multi, x_prop_single, x_zero, x_od_masked, x_dist, mask, pop_raw)
            
            mask_2d = loss_mask.unsqueeze(1) | loss_mask.unsqueeze(2)

            loss_prior = criterion(prior_od, y_od, current_alpha, mask_2d)
            loss_final = criterion(pred_od, y_od, current_alpha, mask_2d)
                
            loss = loss_final + (args.lambda_prior * loss_prior)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{scheduler.get_last_lr()[0]:.1e}"})

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")

        if epoch % 2 == 1 or epoch == args.epochs - 1:
            v_loss, rmse, cpc, rmse_high, cpc_high = validate_mae(model, val_dataset.datasets['seoul'], criterion, device)
            print(f"  ➜ [Val] Loss: {v_loss:.4f} | RMSE: {rmse:.2f} | CPC: {cpc:.4f} | High-Vol RMSE: {rmse_high:.2f} | High-Vol CPC: {cpc_high:.4f}")
            
            if args.use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "val_loss": v_loss,
                    "val_rmse": rmse,
                    "val_cpc": cpc,
                    "val_rmse_high": rmse_high,
                    "val_cpc_high": cpc_high
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
            
        current_dir = os.path.dirname(os.path.abspath(__file__))
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_rmse': best_val_rmse,
            'best_cpc': best_cpc,
            'wandb_run_id': wandb.run.id if (args.use_wandb and wandb.run) else None,
        }, os.path.join(current_dir, 'checkpoint_latest.pth'))

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
