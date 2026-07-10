import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRAIN_CONFIG

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import joblib
import matplotlib.pyplot as plt

from dataset import ODDataset
from models import DeepGravity, SpatialODMAE1, SpatialODMAE5

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

def test_dl_model(args, test_dataset):
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if args.model == 'mae5':
        model = SpatialODMAE5(num_nodes=test_dataset.num_nodes, num_features=test_dataset.X_static.shape[1]).to(device)
    elif args.model == 'mae1':
        model = SpatialODMAE1(num_nodes=test_dataset.num_nodes, num_features=test_dataset.X_static.shape[1]).to(device)
    else:
        model = DeepGravity(num_features=test_dataset.X_static.shape[1]).to(device)
        
    best_model_path = f'best_model_{args.model}.pth'
    
    if not os.path.exists(best_model_path):
        print(f"Error: {best_model_path} not found! Please train the model first.")
        return
        
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    print(f"Loaded {best_model_path} for testing.")
        
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
            
            if args.model != 'mae5' and x_od_masked.ndim == 3:
                x_od_masked = torch.log1p(torch.expm1(x_od_masked).sum(dim=-1))
                y_od = torch.log1p(torch.expm1(y_od).sum(dim=-1))
            
            if args.model in ['mae1', 'mae5']:
                pred = model(x_static, x_od_masked, x_dist, mask)
            else:
                pred = model(x_static, x_dist)
                
            mask_2d = mask.unsqueeze(1) | mask.unsqueeze(2)
            
            if args.model == 'mae5':
                m_exp = mask_2d.unsqueeze(-1).expand_as(y_od)
                loss = weighted_mse_loss(pred[m_exp], y_od[m_exp], alpha=1.0)
                pred_real = torch.expm1(pred[m_exp]).cpu().numpy()
                y_real = torch.expm1(y_od[m_exp]).cpu().numpy()
            else:
                loss = weighted_mse_loss(pred[mask_2d], y_od[mask_2d], alpha=1.0)
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


def test_tabular_model(args, test_dataset):
    X_OD_real = np.expm1(test_dataset.X_OD)
    if X_OD_real.ndim == 3:
        X_OD_real = X_OD_real.sum(axis=-1)
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
    
    save_path = f'best_model_{args.model}.pkl'
    if not os.path.exists(save_path):
        print(f"Error: {save_path} not found! Please train the model first.")
        return
        
    print(f"Loading {save_path}...")
    model = joblib.load(save_path)
    print("Predicting...")
    
    if args.model == 'gravity':
        test_cities = set(test_dataset.test_indices)
        is_train_2d = np.ones((num_nodes, num_nodes), dtype=bool)
        for t in test_cities:
            is_train_2d[t, :] = False
            is_train_2d[:, t] = False
            
        T_pred = model.predict(O_pop_total, D_pop_total, X_dist_real)
        y_pred = T_pred[~is_train_2d]
        y_true = X_OD_real[~is_train_2d]
        
    elif args.model in ['xgb', 'xgb_hurdle']:
        O_stat_feat = X_static[O_idx]
        D_stat_feat = X_static[D_idx]
        X_tabular = np.column_stack([O_pop, D_pop, dist, O_stat_feat, D_stat_feat])
        
        test_cities = set(test_dataset.test_indices)
        is_test = np.array([(o in test_cities) or (d in test_cities) for o, d in zip(O_idx, D_idx)])
        
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=TRAIN_CONFIG['model_type'], choices=['gravity', 'xgb', 'xgb_hurdle', 'deep_gravity', 'mae1', 'mae5'])
    args = parser.parse_args()
    
    channel = 5 if args.model == 'mae5' else 1
    test_dataset = ODDataset(mode='test', channel=channel)
    
    if args.model in ['mae1', 'mae5', 'deep_gravity']:
        test_dl_model(args, test_dataset)
    elif args.model in ['gravity', 'xgb', 'xgb_hurdle']:
        test_tabular_model(args, test_dataset)

if __name__ == '__main__':
    main()