import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRAIN_CONFIG, DATA_DIR

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import ODDataset
from models import DeepGravity, SpatialODMAE1, SpatialODMAE5, XGBGravityModel, GravityModel
from tqdm import tqdm

def main():
    '''
    사용법
    python train.py --model [gravity, xgb, deep_gravity, mae1, mae5] --epochs [NUM_EPOCHS] --batch_size [BATCH_SIZE]
    '''
    # Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=TRAIN_CONFIG['model_type'], choices=['gravity', 'xgb', 'deep_gravity', 'mae1', 'mae5'])
    parser.add_argument('--epochs', type=int, default=TRAIN_CONFIG['epochs'])
    parser.add_argument('--batch_size', type=int, default=TRAIN_CONFIG['batch_size'])
    args = parser.parse_args()
    
    # 데이터셋 로드
    train_dataset = ODDataset(mode='train')
    test_dataset = ODDataset(mode='test')
    
    if args.model in ['mae1', 'mae5', 'deep_gravity']:
        train_dl_model(args, train_dataset, test_dataset)
    elif args.model in ['gravity', 'xgb']:
        train_tabular_model(args, test_dataset)

def train_dl_model(args, train_dataset, test_dataset):
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # 모델 로드
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if args.model == 'mae5':
        model = SpatialODMAE5(num_nodes=train_dataset.num_nodes, num_features=train_dataset.X_static.shape[1]).to(device)
    elif args.model == 'mae1':
        model = SpatialODMAE1(num_nodes=train_dataset.num_nodes, num_features=train_dataset.X_static.shape[1]).to(device)
    else:
        model = DeepGravity(num_features=train_dataset.X_static.shape[1]).to(device)
        
    # Transformer(MAE) 모델의 학습 안정성을 위해 lr을 낮춥니다 (기존 1e-3 -> 1e-4)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    min_mask = TRAIN_CONFIG['min_mask_size']
    max_mask = TRAIN_CONFIG['max_mask_size']
    
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
                
                # 원본 타겟 텐서는 (B, N, N, 5) 이며 값은 log1p 스케일임
                x_od_masked = batch['X_OD_masked'].to(device)
                y_od = batch['y_OD'].to(device)
                
                if args.model != 'mae5':
                    # 1채널 모델용: expm1로 리얼 스케일 복원 -> 5개 채널 합산 -> 다시 log1p 변환
                    x_od_masked = torch.log1p(torch.expm1(x_od_masked).sum(dim=-1))
                    y_od = torch.log1p(torch.expm1(y_od).sum(dim=-1))
                
                optimizer.zero_grad()
                pred = model(x_static, x_od_masked, x_dist, mask)
                    
                mask_2d = mask.unsqueeze(1) | mask.unsqueeze(2)
                
                if args.model == 'mae5':
                    # mask_2d: (B, N, N) -> 타겟 차원은 (B, N, N, 5) 이므로 unsqueeze
                    mask_expanded = mask_2d.unsqueeze(-1).expand_as(y_od)
                    loss = criterion(pred[mask_expanded], y_od[mask_expanded])
                else:
                    loss = criterion(pred[mask_2d], y_od[mask_2d])
                    
                loss.backward()
                
                # 기울기 폭발 방지
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
            print(f"Epoch {epoch+1} Train Loss: {train_loss/len(train_loader):.4f}")
            
            # --- 중간 검증 (Validation) ---
            model.train()
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
                    v_loss = criterion(v_pred[m_exp], y_o[m_exp]).item()
                    p_real = np.maximum(torch.expm1(v_pred[m_exp]).cpu().numpy(), 0)
                    y_real = torch.expm1(y_o[m_exp]).cpu().numpy()
                else:
                    v_loss = criterion(v_pred[m2d], y_o[m2d]).item()
                    p_real = np.maximum(torch.expm1(v_pred[m2d]).cpu().numpy(), 0)
                    y_real = torch.expm1(y_o[m2d]).cpu().numpy()
                    
                rmse = np.sqrt(np.mean((y_real - p_real)**2))
                cpc = cpc_score(y_real, p_real)
            print(f"  ➜ [Val] Loss: {v_loss:.4f} | RMSE: {rmse:.2f} | CPC: {cpc:.4f}")
            model.train()
            
        elif args.model == 'deep_gravity':
            batch = next(iter(train_loader))
            x_static = batch['X_static'].to(device)
            x_dist = batch['X_dist'].to(device)
            
            y_od = batch['y_OD'].to(device)
            # 1채널 모델용 합산 로직
            y_od = torch.log1p(torch.expm1(y_od).sum(dim=-1))
            
            is_train_node = torch.ones(train_dataset.num_nodes, dtype=torch.bool, device=device)
            is_train_node[train_dataset.test_indices] = False
            train_mask_2d = is_train_node.unsqueeze(0) & is_train_node.unsqueeze(1)
            train_mask_2d = train_mask_2d.unsqueeze(0) 
            
            optimizer.zero_grad()
            pred = model(x_static, x_dist)
            
            loss = criterion(pred[train_mask_2d], y_od[train_mask_2d])
            loss.backward()
            optimizer.step()
            
            print(f"Epoch {epoch+1}/{args.epochs} Train Loss: {loss.item():.4f}")
        
    # Testing
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
                loss = criterion(pred[m_exp], y_od[m_exp])
                pred_real = torch.expm1(pred[m_exp]).cpu().numpy()
                y_real = torch.expm1(y_od[m_exp]).cpu().numpy()
            else:
                loss = criterion(pred[mask_2d], y_od[mask_2d])
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
    print(f"Test Loss (MSE log-scale): {test_loss/len(test_loader):.4f}")
    print(f"RMSE (Real scale): {rmse:.2f}")
    print(f"CPC (Common Part of Commuters): {cpc:.4f}")

def train_tabular_model(args, test_dataset):
    # 5개 목적 채널을 마지막 차원에서 합산하여 1채널 2D 매트릭스로 변환
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
        
    elif args.model == 'xgb':
        model = XGBGravityModel()
        O_stat_feat = X_static[O_idx]
        D_stat_feat = X_static[D_idx]
        
        X_tabular = np.column_stack([O_pop, D_pop, dist, O_stat_feat, D_stat_feat])
        
        test_cities = set(test_dataset.test_indices)
        is_test = np.array([(o in test_cities) or (d in test_cities) for o, d in zip(O_idx, D_idx)])
        is_train = ~is_test
        
        print("Fitting XGBoost Model (this might take a while)...")
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

def cpc_score(y_true, y_pred):
    numerator = 2 * np.sum(np.minimum(y_true, y_pred))
    denominator = np.sum(y_true) + np.sum(y_pred)
    if denominator == 0: return 0.0
    return numerator / denominator

if __name__ == '__main__':
    main()