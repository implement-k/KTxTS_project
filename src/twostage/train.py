import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRAIN_CONFIG

import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.validation")

from dataset import ODDataset
from loss import WeightedMSELoss, HybridWeightedMSELoss, HuberLoss
from tqdm import tqdm
from twostage.model import Stage1Model_LGBM, Stage2Model
from validation import validate_twostage

def str2bool(v):
    return str(v).lower() in ("yes", "true", "t", "1")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=TRAIN_CONFIG['epochs'])
    parser.add_argument('--batch_size', type=int, default=TRAIN_CONFIG['batch_size'])
    parser.add_argument('--k_fold', type=int, default=5, help='k-fold 수')
    parser.add_argument('--use_4_lgbm', type=str2bool, default=False, help='nomal, masked 두 경우 다른 모델을 사용할지 선택') 
    parser.add_argument('--use_nan_masking', type=str2bool, default=False, help='NaN 마스킹 사용 여부')
    parser.add_argument('--use_od', type=str2bool, default=False, help='OD 데이터 사용 여부(true면 O,D 예측, false면 inter, self 예측)')
    parser.add_argument('--predict_only_masked', type=str2bool, default=False, help='마스킹된 데이터만 예측할지 여부')
    parser.add_argument('--use_residual', type=str2bool, default=False, help='Residual Learning 적용 여부 (v2)')
    parser.add_argument('--use_wandb', type=str2bool, default=False)
    args = parser.parse_args()
    
    if args.use_wandb: wandb.init(project="TwoStageGravity", config=vars(args))
    
    print("선택된 argument:")
    for arg in vars(args): print(f"  {arg}: {getattr(args, arg)}")
        
    # 데이터셋 로드
    dataset = ODDataset(
        mode='train', 
        use_nan_masking=args.use_nan_masking, 
        use_log_transform=False, 
        use_od=args.use_od, 
        predict_only_masked=args.predict_only_masked,
        use_residual=args.use_residual
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    min_mask = TRAIN_CONFIG['min_mask_size']
    max_mask = TRAIN_CONFIG['max_mask_size']
    
    val_indices = dataset.test_indices
    
    X_static_lgb, y1_train, y2_train, fold_train_mask = dataset.get_stage1_training_data(val_indices)
    
    ################### Stage1: LightGBM PRE-TRAINING ################### 
    print(f"Stage1: LightGBM Pre-training")        
    stage1_model = Stage1Model_LGBM(use_4_lgbm=args.use_4_lgbm) 
    stage1_model.fit_predict(
        X_static=X_static_lgb, 
        y1=np.log1p(y1_train), 
        y2=np.log1p(y2_train),
        masking_indices=dataset.masking_indices
    )
    ###########################################################
    
    ################### Stage2: Two-Stage Gravity Model Training ################### 
    # Configure dataset for this fold
    dataset.train_indices = np.where(fold_train_mask)[0]
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = Stage2Model(
        num_features=dataset.X_static.shape[1], 
        use_od=args.use_od, 
        predict_only_masked=args.predict_only_masked,
        use_residual=args.use_residual
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    total_steps = args.epochs * len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=5e-4, total_steps=total_steps, pct_start=0.3, anneal_strategy='cos'
    )
    
    # mae와 동일하게 loss.py의 Loss 사용
    criterion = WeightedMSELoss().to(device)
    
    # 지표
    best_val_rmse = float('inf')
    best_cpc = 0.0
    
    # Save path can just use best_model_twostage.pth or parameterized name
    best_model_path = f'best_model_twostage_fold.pth'
    
    for epoch in range(args.epochs):
        # mask size 선택
        progress = epoch / max(1, args.epochs - 1)
        current_mask_size = int(min_mask + (max_mask - min_mask) * progress)
        dataset.max_mask_size = current_mask_size
        
        current_alpha = 10.0 - (9.0 * progress)
        
        model.train()
        train_loss = 0
        
        is_train_node = torch.tensor(fold_train_mask, dtype=torch.bool, device=device)
        train_mask_1d = is_train_node
        train_mask_2d = is_train_node.unsqueeze(0) & is_train_node.unsqueeze(1)
        train_mask_2d = train_mask_2d.unsqueeze(0) 
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            x_static = batch['X_static'][0:1].to(device)
            x_dist = batch['X_dist'][0:1].to(device)
            y_od = batch['y_OD'][0:1].to(device)
            y_od_log = torch.log1p(y_od)
            
            x_s_np = x_static.squeeze(0).cpu().numpy()
            log_1_batch, log_2_batch = stage1_model.predict(x_s_np)
            log_1_tensor = torch.tensor(log_1_batch, dtype=torch.float32, device=device).unsqueeze(0)
            log_2_tensor = torch.tensor(log_2_batch, dtype=torch.float32, device=device).unsqueeze(0)
            
            optimizer.zero_grad()
            
            if (args.use_residual or args.predict_only_masked) and not args.use_od:
                v_pred = model(x_static, x_dist, log_1_tensor, log_2_tensor, mask_1d=train_mask_1d, true_OD=y_od)
            else:
                v_pred = model(x_static, x_dist, log_1_tensor, log_2_tensor)

            loss = criterion(v_pred, y_od_log, current_alpha, train_mask_2d)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({'loss': loss.item(), 'lr': f"{current_lr:.1e}"})
            
        avg_train_loss = train_loss/len(train_loader)
        if args.use_wandb:
            wandb.log({f"train_loss": avg_train_loss, f"epoch": epoch})
            
        # === Validation (2 Epoch마다 수행) ===
        if epoch % 2 == 1 or epoch == args.epochs - 1:
            v_loss, rmse, cpc = validate_twostage(
                model=model, 
                stage1_model=stage1_model, 
                dataset=dataset, 
                val_indices=val_indices, 
                criterion=criterion, 
                device=device,
                use_od=args.use_od,
                predict_only_masked=args.predict_only_masked,
                use_residual=args.use_residual
            )
            print(f"  ➜ [Val] Loss: {v_loss:.4f} | RMSE: {rmse:.2f} | CPC: {cpc:.4f}")
            if args.use_wandb:
                wandb.log({f"val_loss": v_loss, f"val_rmse": rmse, f"val_cpc": cpc})
            
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
    
    if args.use_wandb: wandb.finish()
    
if __name__ == '__main__':
    main()
