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
from models import DeepGravity, SpatialODMAE, XGBGravityModel, GravityModel
from tqdm import tqdm

def main():
    '''
    사용법
    python train.py --model [gravity, xgb, deep_gravity, mae] --epochs [NUM_EPOCHS] --batch_size [BATCH_SIZE]
    '''
    # Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=TRAIN_CONFIG['model_type'], choices=['gravity', 'xgb', 'deep_gravity', 'mae'])
    parser.add_argument('--epochs', type=int, default=TRAIN_CONFIG['epochs'])
    parser.add_argument('--batch_size', type=int, default=TRAIN_CONFIG['batch_size'])
    args = parser.parse_args()
    
    # 데이터셋 로드
    train_dataset = ODDataset(DATA_DIR, mode='train')
    test_dataset = ODDataset(DATA_DIR, mode='test')
    
    if args.model in ['mae', 'deep_gravity']:
        train_dl_model(args, train_dataset, test_dataset)
    elif args.model in ['gravity', 'xgb']:
        train_tabular_model(args, test_dataset)

def train_dl_model(args, train_dataset, test_dataset):
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # 모델 로드
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if args.model == 'mae':
        model = SpatialODMAE(num_nodes=train_dataset.num_nodes, num_features=13).to(device)
    else:
        model = DeepGravity(num_features=13).to(device)
        
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    min_mask = TRAIN_CONFIG['min_mask_size']
    max_mask = TRAIN_CONFIG['max_mask_size']
    
    for epoch in range(args.epochs):
        progress = epoch / max(1, args.epochs - 1)
        current_mask_size = int(min_mask + (max_mask - min_mask) * progress)
        train_dataset.max_mask_size = current_mask_size
        
        model.train()
        train_loss = 0
        
        if args.model == 'mae':
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Max Mask: {current_mask_size}]")
            for batch in pbar:
                x_static = batch['X_static'].to(device)
                x_dist = batch['X_distance'].to(device)
                x_od_masked = batch['X_OD_masked'].to(device)
                y_od = batch['y_OD'].to(device)
                mask = batch['mask'].to(device)
                
                optimizer.zero_grad()
                pred = model(x_static, x_od_masked, x_dist, mask)
                    
                mask_2d = mask.unsqueeze(1) | mask.unsqueeze(2)
                loss = criterion(pred[mask_2d], y_od[mask_2d])
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
            print(f"Epoch {epoch+1} Train Loss: {train_loss/len(train_loader):.4f}")
            
        elif args.model == 'deep_gravity':
            # Deep Gravity는 입력이 모든 도시의 static, dist이므로 Batch가 무의미합니다.
            # 전체 Train 도시에 대해 한 번에(Full-batch) 학습합니다.
            batch = next(iter(train_loader)) # 임의의 1개 배치만 가져와서 전체 데이터 활용
            x_static = batch['X_static'].to(device)
            x_dist = batch['X_distance'].to(device)
            y_od = batch['y_OD'].to(device)
            
            # Train 쌍 마스크 (Test 도시가 포함되지 않은 모든 쌍)
            is_train_node = torch.ones(train_dataset.num_nodes, dtype=torch.bool, device=device)
            is_train_node[train_dataset.test_indices] = False
            train_mask_2d = is_train_node.unsqueeze(0) & is_train_node.unsqueeze(1)
            # Batch dimension 맞추기
            train_mask_2d = train_mask_2d.unsqueeze(0) 
            
            optimizer.zero_grad()
            pred = model(x_static, x_dist)
            
            loss = criterion(pred[train_mask_2d], y_od[train_mask_2d])
            loss.backward()
            optimizer.step()
            
            print(f"Epoch {epoch+1}/{args.epochs} Train Loss: {loss.item():.4f}")
        
    # Testing
    model.eval()
    test_loss = 0
    all_y_true = []
    all_y_pred = []
    
    with torch.no_grad():
        for batch in test_loader:
            x_static = batch['X_static'].to(device)
            x_dist = batch['X_distance'].to(device)
            x_od_masked = batch['X_OD_masked'].to(device)
            y_od = batch['y_OD'].to(device)
            mask = batch['mask'].to(device)
            
            if args.model == 'mae':
                pred = model(x_static, x_od_masked, x_dist, mask)
            else:
                pred = model(x_static, x_dist)
                
            mask_2d = mask.unsqueeze(1) | mask.unsqueeze(2)
            loss = criterion(pred[mask_2d], y_od[mask_2d])
            test_loss += loss.item()
            
            pred_real = torch.expm1(pred[mask_2d]).cpu().numpy()
            y_real = torch.expm1(y_od[mask_2d]).cpu().numpy()
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
    X_OD_real = np.expm1(test_dataset.X_OD)
    X_dist_real = np.expm1(test_dataset.X_dist)
    X_static = test_dataset.X_static
    num_nodes = test_dataset.num_nodes
    
    O_idx, D_idx = np.indices((num_nodes, num_nodes))
    O_idx = O_idx.flatten()
    D_idx = D_idx.flatten()
    
    y = X_OD_real.flatten()
    dist = X_dist_real.flatten()
    
    #### TODO x_static의 인구수 데이터 활용해서 O_pop, D_pop 계산 후 모델 학습에 활용
    O_pop_total = X_OD_real.sum(axis=1)
    D_pop_total = X_OD_real.sum(axis=0)
    
    O_pop = O_pop_total[O_idx]
    D_pop = D_pop_total[D_idx]
    
    if args.model == 'gravity':
        # 최적 매개변수 추정
        model = GravityModel()
        
        test_cities = set(test_dataset.test_indices)
        is_train_2d = np.ones((num_nodes, num_nodes), dtype=bool)
        for t in test_cities:
            is_train_2d[t, :] = False
            is_train_2d[:, t] = False
            
        model.fit(O_pop_total, D_pop_total, X_dist_real, X_OD_real, is_train_2d)
        
        print("Predicting...")
        T_pred = model.predict(O_pop_total, D_pop_total, X_dist_real)
        
        # Extract only test pairs
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