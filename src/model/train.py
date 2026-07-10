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
import joblib

from dataset import ODDataset
from models import DeepGravity, SpatialODMAE1, SpatialODMAE5, XGBGravityModel, GravityModel, XGBHurdleModel
from tqdm import tqdm

def main():
    print("Starting Training...")
    '''
    사용법
    python train.py --model [gravity, xgb, xgb_hurdle, deep_gravity, mae1, mae5] --epochs [NUM_EPOCHS] --batch_size [BATCH_SIZE]
    '''
    # Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=TRAIN_CONFIG['model_type'], choices=['gravity', 'xgb', 'xgb_hurdle', 'deep_gravity', 'mae1', 'mae5'])
    parser.add_argument('--epochs', type=int, default=TRAIN_CONFIG['epochs'])
    parser.add_argument('--batch_size', type=int, default=TRAIN_CONFIG['batch_size'])
    args = parser.parse_args()
    
    # 데이터셋 로드
    train_dataset = ODDataset(mode='train')
    test_dataset = ODDataset(mode='test')
    
    if args.model in ['mae1', 'mae5', 'deep_gravity']:
        train_dl_model(args, train_dataset, test_dataset)
    elif args.model in ['gravity', 'xgb', 'xgb_hurdle']:
        train_tabular_model(args, test_dataset)

def weighted_mse_loss(pred, target, alpha=1.5):
    weight = 1.0 + alpha * target
    loss = ((pred - target) ** 2) * weight
    return loss.mean()

def cpc_score(y_true, y_pred):
    numerator = 2 * np.sum(np.minimum(y_true, y_pred))
    denominator = np.sum(y_true) + np.sum(y_pred)
    if denominator == 0: return 0.0
    return numerator / denominator

def train_dl_model(args, train_dataset, test_dataset):
    # dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if args.model == 'mae5':
        model = SpatialODMAE5(num_nodes=train_dataset.num_nodes, num_features=train_dataset.X_static.shape[1]).to(device)
    elif args.model == 'mae1':
        model = SpatialODMAE1(num_nodes=train_dataset.num_nodes, num_features=train_dataset.X_static.shape[1]).to(device)
    else:
        model = DeepGravity(num_features=train_dataset.X_static.shape[1]).to(device)
        
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    total_steps = args.epochs * len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=5e-4,
        total_steps=total_steps,
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    min_mask = TRAIN_CONFIG['min_mask_size']
    max_mask = TRAIN_CONFIG['max_mask_size']
    
    best_val_rmse = float('inf')
    best_model_path = f'best_model_{args.model}.pth'
    
    for epoch in range(args.epochs):
        # masking size 결정(min_mask ~ current_mask_size) 랜덤으로 선택
        progress = epoch / max(1, args.epochs - 1)
        current_mask_size = int(min_mask + (max_mask - min_mask) * progress)
        train_dataset.max_mask_size = current_mask_size
        
        # alpha 값 결정 (10.0 -> 1.0) -> epoch 진행에 따라 감소
        current_alpha = max(1.0, 10.0 * (1.0 - progress))
        
        model.train()
        train_loss = 0
        
        if args.model in ['mae1', 'mae5']:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Mask: {current_mask_size}, α: {current_alpha:.1f}]")
            for batch in pbar:
                x_static = batch['X_static'].to(device)
                x_dist = batch['X_dist'].to(device)
                mask = batch['mask'].to(device)
                
                # mae1: (N,N)
                # mae5: (N,N,5)
                x_od_masked = batch['X_OD_masked'].to(device)
                y_od = batch['y_OD'].to(device)
                
                optimizer.zero_grad()
                pred = model(x_static, x_od_masked, x_dist, mask)
                # mask: (batch, 2N) -> (batch, N, N)
                mask_2d = mask.unsqueeze(1) | mask.unsqueeze(2)
                
                # mae5
                if args.model == 'mae5':
                    # mask: (batch, N, N) -> (batch, N, N, 5)
                    mask_expanded = mask_2d.unsqueeze(-1).expand_as(y_od)
                    loss = weighted_mse_loss(pred[mask_expanded], y_od[mask_expanded], alpha=current_alpha)
                # mae1
                else:
                    loss = weighted_mse_loss(pred[mask_2d], y_od[mask_2d], alpha=current_alpha)
                    
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                current_lr = scheduler.get_last_lr()[0]
                pbar.set_postfix({'loss': loss.item(), 'lr': f"{current_lr:.1e}"})
            print(f"Epoch {epoch+1} Train Loss: {train_loss/len(train_loader):.4f}")
        
        # TODO 개발 필요
        elif args.model == 'deep_gravity':
            batch = next(iter(train_loader))
            x_static = batch['X_static'].to(device)
            x_dist = batch['X_dist'].to(device)
            
            # (N, N)
            y_od = batch['y_OD'].to(device)
            
            is_train_node = torch.ones(train_dataset.num_nodes, dtype=torch.bool, device=device)
            is_train_node[train_dataset.test_indices] = False
            train_mask_2d = is_train_node.unsqueeze(0) & is_train_node.unsqueeze(1)
            train_mask_2d = train_mask_2d.unsqueeze(0) 
            
            optimizer.zero_grad()
            pred = model(x_static, x_dist)
            loss = weighted_mse_loss(pred[train_mask_2d], y_od[train_mask_2d], alpha=current_alpha)
            loss.backward()
            optimizer.step()
            scheduler.step()
            print(f"Epoch {epoch+1}/{args.epochs} Train Loss: {loss.item():.4f}")
            
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
                x_o = val_batch['X_OD_masked'].to(device)
                y_o = val_batch['y_OD'].to(device)
                
                v_pred = model(x_s, x_o, x_d, m)
                m2d = m.unsqueeze(1) | m.unsqueeze(2)
                
                if args.model == 'mae5':
                    m_exp = m2d.unsqueeze(-1).expand_as(y_o)
                    v_loss = weighted_mse_loss(v_pred[m_exp], y_o[m_exp], alpha=1.0).item()
                    p_real = np.maximum(torch.expm1(v_pred[m_exp]).cpu().numpy(), 0)
                    y_real = torch.expm1(y_o[m_exp]).cpu().numpy()
                else:
                    v_loss = weighted_mse_loss(v_pred[m2d], y_o[m2d], alpha=1.0).item()
                    p_real = np.maximum(torch.expm1(v_pred[m2d]).cpu().numpy(), 0)
                    y_real = torch.expm1(y_o[m2d]).cpu().numpy()
                    
                rmse = np.sqrt(np.mean((y_real - p_real)**2))
                cpc = cpc_score(y_real, p_real)
            print(f"  ➜ [Val] Loss: {v_loss:.4f} | RMSE: {rmse:.2f} | CPC: {cpc:.4f}")
            
            if rmse < best_val_rmse:
                best_val_rmse = rmse
                torch.save(model.state_dict(), best_model_path)
                print(f"  ➜ [Checkpoint] Best model saved! (RMSE: {rmse:.2f})")
            
            model.train()
    print("Training finished.")

# TODO 개발 필요
def train_tabular_model(args, test_dataset):
    X_OD_real = np.expm1(test_dataset.X_OD)
    X_dist_real = np.expm1(test_dataset.X_dist)
    X_static = test_dataset.X_static
    num_nodes = test_dataset.num_nodes
    
    O_idx, D_idx = np.indices((num_nodes, num_nodes))
    O_idx = O_idx.flatten()
    D_idx = D_idx.flatten()
    
    y = X_OD_real.flatten()
    dist = X_dist_real.flatten()
    
    O_pop_total = X_OD_real.sum(axis=1)
    D_pop_total = X_OD_real.sum(axis=0)
    O_pop = O_pop_total[O_idx]
    D_pop = D_pop_total[D_idx]
    
    if args.model == 'gravity':
        model = GravityModel()
        test_cities = set(test_dataset.test_indices)
        is_train_2d = np.ones((num_nodes, num_nodes), dtype=bool)
        for t in test_cities:
            is_train_2d[t, :] = False
            is_train_2d[:, t] = False
            
        print(f"Fitting {args.model} Model (this might take a while)...")
        model.fit(O_pop_total, D_pop_total, X_dist_real, X_OD_real, is_train_2d)
        
    elif args.model in ['xgb', 'xgb_hurdle']:
        if args.model == 'xgb_hurdle':
            model = XGBHurdleModel()
        else:
            model = XGBGravityModel()
            
        O_stat_feat = X_static[O_idx]
        D_stat_feat = X_static[D_idx]
        X_tabular = np.column_stack([O_pop, D_pop, dist, O_stat_feat, D_stat_feat])
        
        test_cities = set(test_dataset.test_indices)
        is_test = np.array([(o in test_cities) or (d in test_cities) for o, d in zip(O_idx, D_idx)])
        is_train = ~is_test
        
        print(f"Fitting {args.model} Model (this might take a while)...")
        model.fit(X_tabular[is_train], y[is_train])
    
    # 모델 저장
    save_path = f'best_model_{args.model}.pkl'
    joblib.dump(model, save_path)
    print(f"Training finished. Model saved to {save_path}")

if __name__ == '__main__':
    main()
