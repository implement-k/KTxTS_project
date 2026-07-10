import os
os.environ["OMP_NUM_THREADS"] = "1"
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRAIN_CONFIG

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import ODDataset
from models import DeepGravity, SpatialODMAE1, SpatialODMAE5, XGBGravityModel, GravityModel, XGBHurdleModel
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("Starting main...")
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
    """
    Heavy-tail robust MSE Loss
    target 크기에 비례하여 (1 + alpha * target) 가중치를 부여
    """
    weight = 1.0 + alpha * target
    loss = ((pred - target) ** 2) * weight
    return loss.mean()

def train_dl_model(args, train_dataset, test_dataset):
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
        
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    min_mask = TRAIN_CONFIG['min_mask_size']
    max_mask = TRAIN_CONFIG['max_mask_size']
    
    best_val_rmse = float('inf')
    best_model_path = f'best_model_{args.model}.pth'
    
    for epoch in range(args.epochs):
        progress = epoch / max(1, args.epochs - 1)
        current_mask_size = int(min_mask + (max_mask - min_mask) * progress)
        train_dataset.max_mask_size = current_mask_size
        
        model.train()
        train_loss = 0
        
        if args.model in ['mae1', 'mae5']:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Max Mask: {current_mask_size}]")
            for batch in pbar:
                x_static = batch['X_static'].to(device)
                x_dist = batch['X_dist'].to(device)
                mask = batch['mask'].to(device)
                
                x_od_masked = batch['X_OD_masked'].to(device)
                y_od = batch['y_OD'].to(device)
                
                if args.model != 'mae5':
                    x_od_masked = torch.log1p(torch.expm1(x_od_masked).sum(dim=-1))
                    y_od = torch.log1p(torch.expm1(y_od).sum(dim=-1))
                
                optimizer.zero_grad()
                pred = model(x_static, x_od_masked, x_dist, mask)
                    
                mask_2d = mask.unsqueeze(1) | mask.unsqueeze(2)
                
                if args.model == 'mae5':
                    mask_expanded = mask_2d.unsqueeze(-1).expand_as(y_od)
                    loss = weighted_mse_loss(pred[mask_expanded], y_od[mask_expanded], alpha=1.5)
                else:
                    loss = weighted_mse_loss(pred[mask_2d], y_od[mask_2d], alpha=1.5)
                    
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
            print(f"Epoch {epoch+1} Train Loss: {train_loss/len(train_loader):.4f}")
            
        elif args.model == 'deep_gravity':
            batch = next(iter(train_loader))
            x_static = batch['X_static'].to(device)
            x_dist = batch['X_dist'].to(device)
            
            y_od = batch['y_OD'].to(device)
            y_od = torch.log1p(torch.expm1(y_od).sum(dim=-1))
            
            is_train_node = torch.ones(train_dataset.num_nodes, dtype=torch.bool, device=device)
            is_train_node[train_dataset.test_indices] = False
            train_mask_2d = is_train_node.unsqueeze(0) & is_train_node.unsqueeze(1)
            train_mask_2d = train_mask_2d.unsqueeze(0) 
            
            optimizer.zero_grad()
            pred = model(x_static, x_dist)
            
            loss = weighted_mse_loss(pred[train_mask_2d], y_od[train_mask_2d], alpha=1.5)
            loss.backward()
            optimizer.step()
            
            print(f"Epoch {epoch+1}/{args.epochs} Train Loss: {loss.item():.4f}")
            
        # --- Validation (2 Epoch마다 수행) ---
        if epoch % 2 == 1 or epoch == args.epochs - 1:
            model.train() # PyTorch 버그 우회용
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
                
                if args.model != 'mae5':
                    x_o = torch.log1p(torch.expm1(x_o).sum(dim=-1))
                    y_o = torch.log1p(torch.expm1(y_o).sum(dim=-1))
                
                v_pred = model(x_s, x_o, x_d, m)
                m2d = m.unsqueeze(1) | m.unsqueeze(2)
                
                if args.model == 'mae5':
                    m_exp = m2d.unsqueeze(-1).expand_as(y_o)
                    v_loss = weighted_mse_loss(v_pred[m_exp], y_o[m_exp], alpha=1.5).item()
                    p_real = np.maximum(torch.expm1(v_pred[m_exp]).cpu().numpy(), 0)
                    y_real = torch.expm1(y_o[m_exp]).cpu().numpy()
                else:
                    v_loss = weighted_mse_loss(v_pred[m2d], y_o[m2d], alpha=1.5).item()
                    p_real = np.maximum(torch.expm1(v_pred[m2d]).cpu().numpy(), 0)
                    y_real = torch.expm1(y_o[m2d]).cpu().numpy()
                    
                rmse = np.sqrt(np.mean((y_real - p_real)**2))
                cpc = cpc_score(y_real, p_real)
            print(f"  ➜ [Val] Loss: {v_loss:.4f} | RMSE: {rmse:.2f} | CPC: {cpc:.4f}")
            
            # Checkpoint
            if rmse < best_val_rmse:
                best_val_rmse = rmse
                torch.save(model.state_dict(), best_model_path)
                print(f"  ➜ [Checkpoint] Best model saved! (RMSE: {rmse:.2f})")
            
            model.train()
        
    # Testing
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print("Loaded best model for testing.")
        
    model.train()
    for m_module in model.modules():
        if isinstance(m_module, torch.nn.Dropout):
            m_module.eval()
            
    test_loss = 0
    all_y_true = []
    all_y_pred = []
    
    with torch.no_grad():
        for batch in test_loader:
            x_static = batch['X_static'].to(device)
            x_dist = batch['X_dist'].to(device)
            mask = batch['mask'].to(device)
            
            x_od_masked = batch['X_OD_masked'].to(device)
            y_od = batch['y_OD'].to(device)
            
            if args.model != 'mae5':
                x_od_masked = torch.log1p(torch.expm1(x_od_masked).sum(dim=-1))
                y_od = torch.log1p(torch.expm1(y_od).sum(dim=-1))
            
            if args.model in ['mae1', 'mae5']:
                pred = model(x_static, x_od_masked, x_dist, mask)
            else:
                pred = model(x_static, x_dist)
                
            mask_2d = mask.unsqueeze(1) | mask.unsqueeze(2)
            
            if args.model == 'mae5':
                m_exp = mask_2d.unsqueeze(-1).expand_as(y_od)
                loss = weighted_mse_loss(pred[m_exp], y_od[m_exp])
                pred_real = torch.expm1(pred[m_exp]).cpu().numpy()
                y_real = torch.expm1(y_od[m_exp]).cpu().numpy()
            else:
                loss = weighted_mse_loss(pred[mask_2d], y_od[mask_2d])
                pred_real = torch.expm1(pred[mask_2d]).cpu().numpy()
                y_real = torch.expm1(y_od[mask_2d]).cpu().numpy()
                
            test_loss += loss.item()
            pred_real = np.maximum(pred_real, 0)
            
            all_y_true.append(y_real)
            all_y_pred.append(pred_real)
            
    all_y_true = np.concatenate(all_y_true)
    all_y_pred = np.concatenate(all_y_pred)
    
    rmse = np.sqrt(np.mean((all_y_true - all_y_pred)**2))
    cpc = cpc_score(all_y_true, all_y_pred)
    
    print(f"\n=== Test Results ({args.model}) ===")
    print(f"Test Loss (Weighted MSE log-scale): {test_loss/len(test_loader):.4f}")
    print(f"RMSE (Real scale): {rmse:.2f}")
    print(f"CPC (Common Part of Commuters): {cpc:.4f}")
    
    visualize_predictions(all_y_true, all_y_pred, args.model)


def train_tabular_model(args, test_dataset):
    X_OD_real = np.expm1(test_dataset.X_OD).sum(axis=-1)
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
            
        model.fit(O_pop_total, D_pop_total, X_dist_real, X_OD_real, is_train_2d)
        print("Predicting...")
        T_pred = model.predict(O_pop_total, D_pop_total, X_dist_real)
        y_pred = T_pred[~is_train_2d]
        y_true = X_OD_real[~is_train_2d]
        
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
        print("Predicting...")
        y_pred = model.predict(X_tabular[is_test])
        y_pred = np.maximum(y_pred, 0)
        y_true = y[is_test]
    
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    cpc = cpc_score(y_true, y_pred)
    
    print(f"\n=== Test Results ({args.model}) ===")
    print(f"Test size: {len(y_true)} OD pairs")
    print(f"RMSE (Real scale): {rmse:.2f}")
    print(f"CPC (Common Part of Commuters): {cpc:.4f}")
    
    visualize_predictions(y_true, y_pred, args.model)


def visualize_predictions(y_true, y_pred, model_name):
    plt.figure(figsize=(12, 5))
    
    # Scatter Plot
    plt.subplot(1, 2, 1)
    # 0값 처리를 위해 log1p 사용
    plt.scatter(np.log1p(y_true), np.log1p(y_pred), alpha=0.3, s=2)
    plt.plot([0, 10], [0, 10], 'r--')
    plt.xlabel('True OD (log1p)')
    plt.ylabel('Predicted OD (log1p)')
    plt.title(f'Scatter Plot ({model_name})')
    
    # Residual Plot
    plt.subplot(1, 2, 2)
    residual = y_pred - y_true
    plt.scatter(np.log1p(y_true), residual, alpha=0.3, s=2)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel('True OD (log1p)')
    plt.ylabel('Residual (Pred - True)')
    plt.title('Residual Plot (Heavy-tail Check)')
    
    plt.tight_layout()
    save_path = f'results_{model_name}.png'
    plt.savefig(save_path)
    plt.close()
    print(f"Visualization saved to {save_path}")


def cpc_score(y_true, y_pred):
    numerator = 2 * np.sum(np.minimum(y_true, y_pred))
    denominator = np.sum(y_true) + np.sum(y_pred)
    if denominator == 0: return 0.0
    return numerator / denominator

if __name__ == '__main__':
    main()

