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

from dataset import ODDataset
from loss import WeightedMSELossWrapper
from tqdm import tqdm
from twostage.model import Stage1Model
from twostage.model import Stage2Model

def main():
    print("Two-Stage model")
    '''
    사용법
    python train.py --epochs [NUM_EPOCHS] --batch_size [BATCH_SIZE]
    '''
    # Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=TRAIN_CONFIG['epochs'])
    parser.add_argument('--batch_size', type=int, default=TRAIN_CONFIG['batch_size'])
    args = parser.parse_args()
    
    # 데이터셋 로드
    train_dataset = ODDataset(mode='train')
    test_dataset = ODDataset(mode='test')
    
    # dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    ################### Stage1: LightGBM PRE-TRAINING ################### 
    print("Stage1: LightGBM Pre-training")        
    stage1_model = Stage1Model()
    
    # x_static만 보고 y_self, y_inter를 예측하도록 학습
    print("Fitting LGBM")
    log_self_all, log_inter_all = stage1_model.fit_predict(
        X_static = train_dataset.X_static_lgb, 
        y_self=np.log1p(train_dataset.y_self_train), 
        y_inter=np.log1p(train_dataset.y_inter_train)
        )
    
    # Convert to tensors
    log_self_tensor = torch.tensor(log_self_all, dtype=torch.float32, device=device).unsqueeze(0) # (1, N)
    log_inter_tensor = torch.tensor(log_inter_all, dtype=torch.float32, device=device).unsqueeze(0) # (1, N)
    print("LightGBM Stage 1 Finished!")
    ###########################################################

    ################### Stage2: Two-Stage Gravity Model Training ################### 
    model = Stage2Model(num_features=train_dataset.X_static.shape[1]).to(device)
        
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    total_steps = args.epochs * len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=5e-4,
        total_steps=total_steps,
        pct_start=0.3,
        anneal_strategy='cos'
    )
    criterion = WeightedMSELossWrapper(alpha=10.0).to(device)
    
    min_mask = TRAIN_CONFIG['min_mask_size']
    max_mask = TRAIN_CONFIG['max_mask_size']
    
    best_val_rmse = float('inf')
    best_model_path = f'best_model_twostage.pth'
    
    for epoch in range(args.epochs):
        # masking size 결정(min_mask ~ current_mask_size) 랜덤으로 선택
        progress = epoch / max(1, args.epochs - 1)
        current_mask_size = int(min_mask + (max_mask - min_mask) * progress)
        train_dataset.max_mask_size = current_mask_size
        
        # Alpha decay (10.0 -> 1.0)
        current_alpha = 10.0 - (9.0 * progress)
        criterion.alpha = current_alpha
        
        model.train()
        train_loss = 0
        
        is_train_node = torch.ones(train_dataset.num_nodes, dtype=torch.bool, device=device)
        is_train_node[train_dataset.test_indices] = False
        train_mask_2d = is_train_node.unsqueeze(0) & is_train_node.unsqueeze(1)
        train_mask_2d = train_mask_2d.unsqueeze(0) 
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            # Same memory-optimization technique as deep_gravity
            x_static = batch['X_static'][0:1].to(device)
            x_dist = batch['X_dist'][0:1].to(device)
            y_od = batch['y_OD'][0:1].to(device)
            y_od_log = torch.log1p(y_od)
            
            optimizer.zero_grad()
            v_pred = model(x_static, x_dist, log_self_tensor, log_inter_tensor)

            loss = criterion(v_pred, y_od_log, train_mask_2d)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({'loss': loss.item(), 'lr': f"{current_lr:.1e}"})
            
        print(f"Epoch {epoch+1} Train Loss: {train_loss/len(train_loader):.4f}")
            
        # === Validation (2 Epoch마다 수행) ===
        if epoch % 2 == 1 or epoch == args.epochs - 1:
            model.train() # PyTorch 우회용
            for m_module in model.modules():
                if isinstance(m_module, torch.nn.Dropout):
                    m_module.eval()
            with torch.no_grad():
                val_batch = next(iter(test_loader))
                x_s = val_batch['X_static'].to(device)
                x_d = val_batch['X_dist'].to(device)
                m = val_batch['mask'].to(device)
                y_o = val_batch['y_OD'].to(device)
                y_o_log = torch.log1p(y_o)
            
                v_pred = model(x_s, x_d, log_self_tensor, log_inter_tensor)

                m2d = m.unsqueeze(1) | m.unsqueeze(2)
                
                v_loss = criterion(v_pred, y_o_log, m2d).item()
                p_real = np.maximum(torch.expm1(v_pred[m2d]).cpu().numpy(), 0)
                y_real = y_o[m2d].cpu().numpy()
                    
                rmse = np.sqrt(np.mean((y_real - p_real)**2))
                cpc = cpc_score(y_real, p_real)
            print(f"  ➜ [Val] Loss: {v_loss:.4f} | RMSE: {rmse:.2f} | CPC: {cpc:.4f}")
            
            if rmse < best_val_rmse:
                best_val_rmse = rmse
                torch.save(model.state_dict(), best_model_path)
                print(f"  ➜ [Checkpoint] Best model saved! (RMSE: {rmse:.2f})")
            
            model.train()
    print("Training finished.")
    ###########################################################
    

def cpc_score(y_true, y_pred):
    numerator = 2 * np.sum(np.minimum(y_true, y_pred))
    denominator = np.sum(y_true) + np.sum(y_pred)
    if denominator == 0: return 0.0
    return numerator / denominator
    
if __name__ == '__main__':
    main()
