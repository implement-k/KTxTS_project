import os
import sys
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

current_dir = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(current_dir)
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from dataset import ODDataset

def cpc_score(y_true, y_pred):
    numerator = 2 * np.sum(np.minimum(y_true, y_pred))
    denominator = np.sum(y_true) + np.sum(y_pred)
    if denominator == 0:
        return 0.0
    return numerator / denominator

def make_pairs(dataset, include_self_loop=False):
    # Vectorized extraction of Train Pairs
    train_idx = dataset.train_indices
    
    I, J = np.meshgrid(train_idx, train_idx, indexing='ij')
    
    if not include_self_loop:
        mask = I != J
        I = I[mask]
        J = J[mask]
    else:
        I = I.flatten()
        J = J.flatten()
        
    O_feats = dataset.X_static[I]
    D_feats = dataset.X_static[J]
    Dist = dataset.X_dist[I, J].reshape(-1, 1)
    Target = dataset.X_OD[I, J]
    
    X = np.concatenate([O_feats, D_feats, Dist], axis=1)
    y = Target
    
    return X, y, I, J

def main():
    print("Dataset 로딩 중 (2019 & 2023)...")
    dataset_19 = ODDataset(mode='train', year='2019')
    dataset_23 = ODDataset(mode='train', year='2023')
    
    print("\nTraining 데이터 (O-D Pairs) 구축 중...")
    X_19, y_19, _, _ = make_pairs(dataset_19, include_self_loop=False)
    X_23, y_23, _, _ = make_pairs(dataset_23, include_self_loop=False)
    
    X_train = np.concatenate([X_19, X_23], axis=0)
    y_train = np.concatenate([y_19, y_23], axis=0)
    
    print(f"총 통합 학습 데이터 크기: X={X_train.shape}, y={y_train.shape}")
    
    lgb_train = lgb.Dataset(X_train, y_train)
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.1,
        'num_leaves': 63,
        'verbose': -1,
        'n_jobs': -1
    }
    
    print("LGBM Gravity 모델 훈련 진행 중 (2019+2023)...")
    lgbm_model = lgb.train(params, lgb_train, num_boost_round=150)
    
    # ---------------- Validation ----------------
    print("\nValidation 진행 중 (2023년 데이터 기준)...")
    val_idx = dataset_23.val_indices
    all_idx = dataset_23.all_indices
    
    # val -> all
    I_val, J_val = np.meshgrid(val_idx, all_idx, indexing='ij')
    # all -> val
    I_all, J_val2 = np.meshgrid(all_idx, val_idx, indexing='ij')
    
    pairs_val = set(zip(I_val.flatten(), J_val.flatten()))
    pairs_val.update(zip(I_all.flatten(), J_val2.flatten()))
    pairs_val = np.array([p for p in pairs_val if p[0] != p[1]])
    
    I_v = pairs_val[:, 0]
    J_v = pairs_val[:, 1]
    
    X_val = np.concatenate([
        dataset_23.X_static[I_v], 
        dataset_23.X_static[J_v], 
        dataset_23.X_dist[I_v, J_v].reshape(-1, 1)
    ], axis=1)
    
    y_val_true = dataset_23.X_OD[I_v, J_v]
    y_val_pred = lgbm_model.predict(X_val)
    
    # 변환된 log1p 값을 원래대로 복구
    y_val_true_raw = np.expm1(y_val_true)
    y_val_pred_raw = np.maximum(np.expm1(y_val_pred), 0)
    
    rmse = np.sqrt(mean_squared_error(y_val_true_raw, y_val_pred_raw))
    cpc = cpc_score(y_val_true_raw, y_val_pred_raw)
    
    print(f"Validation RMSE: {rmse:.4f}")
    print(f"Validation CPC:  {cpc:.4f}")
    
    # 모델 저장
    best_model_dir = os.path.join(BASE_DIR, '../best_model')
    os.makedirs(best_model_dir, exist_ok=True)
    lgbm_path = os.path.join(best_model_dir, 'lgbm_gravity_combined.txt')
    lgbm_model.save_model(lgbm_path)
    
    print(f"\n✅ LGBM 통합 Gravity 모델 저장 완료: {lgbm_path}")

if __name__ == '__main__':
    main()
