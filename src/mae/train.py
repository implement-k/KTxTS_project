import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRAIN_CONFIG

import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import MultiRegionDataset
from mae.models import SpatialODMAE
from tqdm import tqdm
try:
    import wandb
except ImportError:
    wandb = None
from loss import WeightedMSELoss, HybridWeightedMSELoss, HuberLoss
from validation import validate_mae
import lightgbm as lgb
import numpy as np

def str2bool(v):
    return str(v).lower() in ("yes", "true", "t", "1")

def main():
    print("test v8")
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=TRAIN_CONFIG['epochs'])
    parser.add_argument('--batch_size', type=int, default=TRAIN_CONFIG['batch_size'])
    parser.add_argument('--loss_type', type=str, default='weighted_mse', choices=['weighted_mse', 'hybrid', 'huber']) # v1, v2, v3: weighted_mse
    parser.add_argument('--use_friction', type=str2bool, default=True)              # v1, v2, v3: False, v4: True
    parser.add_argument('--use_self_loop_predictor', type=str2bool, default=True)   # v1: False, v2, v3, v4: True
    parser.add_argument('--lambda_diag', type=float, default=1.0)                   # v6: 50(수치상으로는 130이 맞긴함)
    parser.add_argument('--use_lgbm_self_loop', type=str2bool, default=False)       # v7: True
    parser.add_argument('--use_wandb', type=str2bool, default=False)
    args = parser.parse_args()
    
    if args.use_wandb: wandb.init(project="SpatialODMAE", config=vars(args))
    
    print("선택된 argument:")
    for arg in vars(args): print(f"  {arg}: {getattr(args, arg)}")

    # 사용 가능한 지역들 추가
    regions = ['seoul', 'jeju', 'busan', 'daegu', 'daejeon', 'gwangju']
    dataset = MultiRegionDataset(regions=regions, batch_size=args.batch_size, mode='train')
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps'  if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    min_mask = TRAIN_CONFIG['min_mask_size']
    max_mask = TRAIN_CONFIG['max_mask_size']

    # Validation 대상은 Test 도시 전체 (MultiRegionDataset의 seoul dataset에서 추출)
    val_indices = dataset.datasets['seoul'].test_indices
    # Dataset 내부에서 batch_size를 처리하므로 DataLoader는 batch_size=1로 둔다.
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # 모델 초기화 (seoul의 정적 피처 개수 기준)
    model = SpatialODMAE(num_features=dataset.datasets['seoul'].X_static.shape[1],
                         use_distance_friction=args.use_friction,
                         use_self_loop_predictor=args.use_self_loop_predictor).to(device)
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
    else: criterion = WeightedMSELoss().to(device)

    best_val_rmse = float('inf')
    best_cpc = 0.0
    best_model_path = 'best_model_mae.pth'

    for epoch in range(args.epochs):
        progress = epoch / max(1, args.epochs - 1)
        current_mask_size = int(min_mask + (max_mask - min_mask) * progress)
        dataset.max_mask_size = current_mask_size
        current_alpha = max(1.0, 10.0 * (1.0 - progress))

        model.train()
        train_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} " f"[Mask:{current_mask_size} α:{current_alpha:.1f}]")
        for step, batch in enumerate(train_loader):
            # DataLoader가 반환한 값은 (1, B, N, ...) 형태이므로 squeeze(0) 수행
            x_static = batch['X_static'].squeeze(0).to(device)
            x_dist = batch['X_dist'].squeeze(0).to(device)
            x_od_masked = batch['X_OD_masked'].squeeze(0).to(device)
            y_od = batch['y_OD'].squeeze(0).to(device)
            mask = batch['mask'].squeeze(0).to(device)
            has_static = batch['has_static'].squeeze(0).to(device)
            
            optimizer.zero_grad()
            pred_od, pred_static = model(x_static, x_od_masked, x_dist, mask)
            
            diag_mask = torch.eye(dataset.num_nodes, device=device, dtype=torch.bool).unsqueeze(0).expand(pred_od.shape[0], -1, -1)
            mask_2d = mask.unsqueeze(1) | mask.unsqueeze(2)

            if args.lambda_diag < 0:
                loss_od = criterion(pred_od, y_od, current_alpha, mask_2d)
            else:
                valid_diag_mask = diag_mask & mask_2d
                valid_offdiag_mask = (~diag_mask) & mask_2d
                
                loss_diag = criterion(pred_od, y_od, current_alpha, valid_diag_mask) if valid_diag_mask.any() else 0.0
                loss_offdiag = criterion(pred_od, y_od, current_alpha, valid_offdiag_mask) if valid_offdiag_mask.any() else 0.0
                
                loss_od = loss_offdiag + (args.lambda_diag * loss_diag)
                
            # Static Feature Loss (MSE on masked nodes) - Only for regions with static features (Seoul)
            if mask.any() and has_static.any():
                loss_static = torch.nn.functional.mse_loss(pred_static[mask], x_static[mask])
            else:
                loss_static = 0.0
                
            loss = loss_od + loss_static

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
            # validation은 seoul 데이터셋에 대해서만 평가
            v_loss, rmse, cpc = validate_mae(model, dataset.datasets['seoul'], val_indices, criterion, device)
            print(f"  ➜ [Val] Loss: {v_loss:.4f} | RMSE: {rmse:.2f} | CPC: {cpc:.4f}")
            
            if args.use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "val_loss": v_loss,
                    "val_rmse": rmse,
                    "val_cpc": cpc
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
