import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader
import joblib
import matplotlib.pyplot as plt
from twostage.model import Stage2Model, Stage1Model_DualLGBM
from dataset import ODDataset

def cpc_score(y_true, y_pred):
    numerator = 2 * np.sum(np.minimum(y_true, y_pred))
    denominator = np.sum(y_true) + np.sum(y_pred)
    if denominator == 0: return 0.0
    return numerator / denominator

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

def weighted_mse_loss(pred, target, alpha=1.5):
    weight = 1.0 + alpha * target
    loss = ((pred - target) ** 2) * weight
    return loss.mean()

def main():    
    test_dataset = ODDataset(mode='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = Stage2Model(num_features=test_dataset.X_static.shape[1]).to(device)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    best_model_path = os.path.join(current_dir, 'best_model_twostage_3_fold_1.pth')
    
    if not os.path.exists(best_model_path):
        print(f"Error: {best_model_path} not found! Please train the model first.")
        return
        
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    print(f"Loaded {best_model_path} for testing.")
        
    model.train()
    for m_module in model.modules():
        if isinstance(m_module, torch.nn.Dropout):
            m_module.eval()
            
    stage1 = Stage1Model_DualLGBM()
    # Test 시에는 편의상 첫 번째 Fold의 모델을 사용합니다.
    # K-Fold Ensemble을 하려면 각 Fold별 예측값의 평균을 내야 합니다.
    stage1.normal_self = joblib.load(os.path.join(current_dir, 'lgbm_normal_self_fold_1.pkl'))
    stage1.normal_inter = joblib.load(os.path.join(current_dir, 'lgbm_normal_inter_fold_1.pkl'))
    stage1.masked_self = joblib.load(os.path.join(current_dir, 'lgbm_masked_self_fold_1.pkl'))
    stage1.masked_inter = joblib.load(os.path.join(current_dir, 'lgbm_masked_inter_fold_1.pkl'))
    
    log_self_all, log_inter_all = stage1.predict(test_dataset.X_static)
    log_self_tensor = torch.tensor(log_self_all, dtype=torch.float32, device=device).unsqueeze(0)
    log_inter_tensor = torch.tensor(log_inter_all, dtype=torch.float32, device=device).unsqueeze(0)

    test_loss = 0
    all_y_true = []
    all_y_pred = []
    
    with torch.no_grad():
        for batch in test_loader:
            x_static = batch['X_static'].to(device)
            x_dist = batch['X_dist'].to(device)
            mask = batch['mask'].to(device)
            
            y_od = batch['y_OD'].to(device)
            y_od_log = torch.log1p(y_od)
            
            pred = model(x_static, x_dist, log_self_tensor, log_inter_tensor)
                
            mask_2d = mask.unsqueeze(1) | mask.unsqueeze(2)
            
            loss = weighted_mse_loss(pred[mask_2d], y_od_log[mask_2d], alpha=1.0)
            pred_real = torch.expm1(pred[mask_2d]).cpu().numpy()
            y_real = y_od[mask_2d].cpu().numpy()
                
            test_loss += loss.item()
            pred_real = np.maximum(pred_real, 0)
            
            all_y_true.append(y_real)
            all_y_pred.append(pred_real)
            
    all_y_true = np.concatenate(all_y_true)
    all_y_pred = np.concatenate(all_y_pred)
    
    rmse = np.sqrt(np.mean((all_y_true - all_y_pred)**2))
    cpc = cpc_score(all_y_true, all_y_pred)
    
    print(f"\n=== Test Results (twostage) ===")
    print(f"Test Loss (Weighted MSE log-scale): {test_loss/len(test_loader):.4f}")
    print(f"RMSE (Real scale): {rmse:.2f}")
    print(f"CPC (Common Part of Commuters): {cpc:.4f}")
    
    visualize_predictions(all_y_true, all_y_pred, "twostage")
    
    import pandas as pd
    if 'pred' in locals():
        pred_full_real = torch.expm1(pred[0]).cpu().numpy()
        pred_full_real = np.maximum(pred_full_real, 0)
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        dong_path = os.path.join(current_dir, '..', '..', 'dataset', 'raw', 'OD_dong_list.xlsx')
        dong_df = pd.read_excel(dong_path)
        dongs = dong_df['dong_code'].values
        
        df_pred = pd.DataFrame(pred_full_real, index=dongs, columns=dongs)
        df_pred.to_csv(os.path.join(current_dir, "predicted_twostage_matrix.csv"), index=True)
        print("Saved full predicted OD matrix to predicted_twostage_matrix.csv")

if __name__ == '__main__':
    main()