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
import random

from twostage.dataset import ODDataset
from loss import WeightedMSELossWrapper
from tqdm import tqdm
from twostage.model import Stage1Model_LGBM
from twostage.model import Stage2Model

def create_k_folds(train_indices, X_dist, k_fold=5):
    # train_indices: available nodes for training/val (excluding test cities)
    unassigned = set(train_indices)
    val_cities = []
    
    while len(unassigned) > 0:
        if len(unassigned) <= 8:
            val_cities.append(list(unassigned))
            break
            
        val_city_center = random.choice(list(unassigned))
        val_city_size = random.randint(4, 8)
        
        # validation city center와 unassigned nodes 간의 거리 계산
        unassigned_list = list(unassigned)
        dists = X_dist[val_city_center, unassigned_list]
        
        # validation city size만큼 가장 가까운 노드 선택
        nearest_idx = np.argsort(dists)[:val_city_size]
        val_city = [unassigned_list[i] for i in nearest_idx]
        
        # validation city 추가 및 unassigned에서 제거
        val_cities.append(val_city)
        for dong in val_city:
            unassigned.remove(dong)
            
    # shuffle하고 fold로 나누기
    random.shuffle(val_cities)
    folds = [[] for _ in range(k_fold)]
    for i, val_city in enumerate(val_cities):
        folds[i % k_fold].extend(val_city)
        
    return [np.array(f, dtype=int) for f in folds]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=TRAIN_CONFIG['epochs'])
    parser.add_argument('--batch_size', type=int, default=TRAIN_CONFIG['batch_size'])
    parser.add_argument('--k_fold', type=int, default=5, help='k-fold 수')
    args = parser.parse_args()
    
    # 데이터셋 로드
    dataset = ODDataset(mode='train')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # fold 생성
    folds = create_k_folds(dataset.train_indices, dataset.X_dist, k_fold=args.k_fold)
    
    cv_rmses = []
    cv_cpcs = []
    
    min_mask = TRAIN_CONFIG['min_mask_size']
    max_mask = TRAIN_CONFIG['max_mask_size']
    
    for fold in range(args.k_fold):
        print(f"\n{'='*20} FOLD {fold+1}/{args.k_fold} {'='*20}")
        val_indices = folds[fold]
        
        # validation, test 도시 마스킹한 데이터셋 생성
        X_static_lgb, y_self_train, y_inter_train, fold_train_mask = dataset.get_stage1_training_data(val_indices)
        
        ################### Stage1: LightGBM PRE-TRAINING ################### 
        print(f"Stage1: LightGBM Pre-training (Fold {fold+1})")        
        stage1_model = Stage1Model_LGBM() 
        stage1_model.fit_predict(
            X_static=X_static_lgb, 
            y_self=np.log1p(y_self_train), 
            y_inter=np.log1p(y_inter_train),
            masking_indices=dataset.masking_indices,
            fold=fold+1
        )
        ###########################################################

        ################### Stage2: Two-Stage Gravity Model Training ################### 
        # Configure dataset for this fold
        dataset.train_indices = np.where(fold_train_mask)[0]
        train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        model = Stage2Model(num_features=dataset.X_static.shape[1]).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        total_steps = args.epochs * len(train_loader)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=5e-4, total_steps=total_steps, pct_start=0.3, anneal_strategy='cos'
        )
        criterion = WeightedMSELossWrapper(alpha=10.0).to(device)
        
        # 지표
        best_val_rmse = float('inf')
        best_cpc = 0.0
        best_model_path = f'best_model_twostage_3_fold_{fold+1}.pth'
        
        for epoch in range(args.epochs):
            # mask size 선택 (min_mask ~ current_mask_size 중에 랜덤으로 선택함)
            progress = epoch / max(1, args.epochs - 1)
            current_mask_size = int(min_mask + (max_mask - min_mask) * progress)
            dataset.max_mask_size = current_mask_size
            
            # Alpha decay (10.0 -> 1.0)
            current_alpha = 10.0 - (9.0 * progress)
            criterion.alpha = current_alpha
            
            model.train()
            train_loss = 0
            
            is_train_node = torch.tensor(fold_train_mask, dtype=torch.bool, device=device)
            train_mask_2d = is_train_node.unsqueeze(0) & is_train_node.unsqueeze(1)
            train_mask_2d = train_mask_2d.unsqueeze(0) 
            
            pbar = tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch+1}/{args.epochs}")
            for batch in pbar:
                x_static = batch['X_static'][0:1].to(device)
                x_dist = batch['X_dist'][0:1].to(device)
                y_od = batch['y_OD'][0:1].to(device)
                y_od_log = torch.log1p(y_od)
                
                # Dynamically predict Stage 1 outputs using the currently MASKED static features
                x_s_np = x_static.squeeze(0).cpu().numpy()
                log_self_batch, log_inter_batch = stage1_model.predict(x_s_np)
                log_self_tensor = torch.tensor(log_self_batch, dtype=torch.float32, device=device).unsqueeze(0)
                log_inter_tensor = torch.tensor(log_inter_batch, dtype=torch.float32, device=device).unsqueeze(0)
                
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
                model.eval()
                with torch.no_grad():
                    # 2. Validation Data Prep (Dataset 내부 로직 사용)
                    X_static_masked, x_s, x_d, y_o, y_o_log, val_mask_2d = dataset.get_validation_data(val_indices)
                    x_s, x_d = x_s.to(device), x_d.to(device)
                    y_o, y_o_log = y_o.to(device), y_o_log.to(device)
                    val_mask_2d = val_mask_2d.to(device)
                    
                    # Dynamically predict Stage 1 outputs for validation using MASKED static features
                    log_self_val, log_inter_val = stage1_model.predict(X_static_masked)
                    log_self_tensor = torch.tensor(log_self_val, dtype=torch.float32, device=device).unsqueeze(0)
                    log_inter_tensor = torch.tensor(log_inter_val, dtype=torch.float32, device=device).unsqueeze(0)
                
                    v_pred = model(x_s, x_d, log_self_tensor, log_inter_tensor)

                    v_loss = criterion(v_pred, y_o_log, val_mask_2d).item()
                    p_real = np.maximum(torch.expm1(v_pred[val_mask_2d]).cpu().numpy(), 0)
                    y_real = y_o[val_mask_2d].cpu().numpy()
                        
                    rmse = np.sqrt(np.mean((y_real - p_real)**2))
                    cpc = cpc_score(y_real, p_real)
                print(f"  ➜ [Val] Loss: {v_loss:.4f} | RMSE: {rmse:.2f} | CPC: {cpc:.4f}")
                
                if rmse < best_val_rmse:
                    best_val_rmse = rmse
                    best_cpc = cpc
                    torch.save(model.state_dict(), best_model_path)
                    print(f"  ➜ [Checkpoint] Best model saved! (RMSE: {rmse:.2f})")
                
                model.train()
        
        cv_rmses.append(best_val_rmse)
        cv_cpcs.append(best_cpc)
        print(f"Fold {fold+1} finished.")
        ###########################################################
        
    print(f"\n========== K-FOLD CV RESULTS ==========")
    for i, (r, c) in enumerate(zip(cv_rmses, cv_cpcs)):
        print(f"Fold {i+1} Best RMSE: {r:.2f} | CPC: {c:.4f}")
    print(f"Average RMSE: {np.mean(cv_rmses):.2f} | Average CPC: {np.mean(cv_cpcs):.4f}")
    print("=======================================")

def cpc_score(y_true, y_pred):
    numerator = 2 * np.sum(np.minimum(y_true, y_pred))
    denominator = np.sum(y_true) + np.sum(y_pred)
    if denominator == 0: return 0.0
    return numerator / denominator
    
if __name__ == '__main__':
    main()
