import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any

# Setup paths
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC_DIR)

from config import DATA_DIR, STATIC_DATA_23_PATH, MASKING_COLUMNS
from mae_year.models import ODMAE # or SpatialODMAE

def get_static_feature_names():
    # Load static features to get names
    static_df = pd.read_csv(STATIC_DATA_23_PATH)
    feature_cols = [c for c in static_df.columns if c not in ['dong_code', 'dong_name']]
    feature_cols = sorted(feature_cols)
    # Plus indicator cols added in dataset
    feature_cols.extend(['is_masked', 'is_merged'])
    return feature_cols

def explain_model(ckpt_path: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load test dataset
    test_data_path = os.path.join(DATA_DIR, 'fixed_eval/fixed_test_dataset.pt')
    if not os.path.exists(test_data_path):
        print("Error: Test dataset not found.")
        return
    test_data = torch.load(test_data_path, map_location='cpu', weights_only=False)
    
    # Pick a sample, e.g. '동탄'
    city = '동탄'
    task_id = 1 # Task 1 is typical random masking
    sample = test_data[city][task_id][0]
    
    N = sample['X_static'].shape[0]
    num_features = sample['X_static'].shape[1]
    
    # Initialize model
    try:
        model = ODMAE(num_features=num_features, d_model=128, num_layers=4, nhead=8)
    except Exception as e:
        print(f"Failed to load ODMAE: {e}")
        from mae_year.models import SpatialODMAE
        model = SpatialODMAE(num_nodes=N, num_features=num_features, d_model=128, num_layers=4, nhead=8)
    
    print(f"Loading weights from {ckpt_path}...")
    try:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    except Exception as e:
        print(f"Load failed, attempting strict=False... {e}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)
    
    model.to(device)
    model.eval()

    # Move inputs to device and set requires_grad
    x_static = sample['X_static'].unsqueeze(0).to(device)
    x_static.requires_grad_(True)
    x_dist = sample['X_dist'].unsqueeze(0).to(device)
    x_dist.requires_grad_(True)
    x_od_masked = sample['X_OD_masked'].unsqueeze(0).to(device)
    A_spatial = sample['A_spatial'].unsqueeze(0).to(device)
    mask = sample['mask'].unsqueeze(0).to(device)
    active_node_mask = sample['active_node_mask'].unsqueeze(0).to(device) if 'active_node_mask' in sample else None
    
    # Setup Hooks for intermediate values
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    # Hook for gate value and attention weights
    if hasattr(model, 'od_gate'):
        model.od_gate.register_forward_hook(get_activation('gate_val'))
    
    # Forward pass
    with torch.set_grad_enabled(True):
        if active_node_mask is not None:
            pred_od = model(x_static, x_od_masked, x_dist, A_spatial, mask, active_node_mask)
        else:
            pred_od = model(x_static, x_od_masked, x_dist, A_spatial, mask)
        
        # Target node prediction sum as target for gradients
        # We focus on the masked nodes' predictions
        target_pred = pred_od[0, mask[0], :].sum() + pred_od[0, :, mask[0]].sum()
        
        # Backward pass
        target_pred.backward()

    # 1. Feature Importance (Input * Gradient)
    static_grad = x_static.grad[0].cpu().numpy() # (N, F)
    static_val = x_static[0].detach().cpu().numpy()
    importance = np.abs(static_grad * static_val).mean(axis=0)
    
    feature_names = get_static_feature_names()
    print("\n=== Static Feature Importance ===")
    feat_imp = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)
    for name, imp in feat_imp:
        print(f"{name}: {imp:.4f}")
        
    dist_grad = x_dist.grad[0].cpu().numpy()
    dist_val = x_dist[0].detach().cpu().numpy()
    dist_importance = np.abs(dist_grad * dist_val).mean()
    print(f"\nDistance Feature Importance: {dist_importance:.4f}")

    # 2. Gate Value (Influence of surrounding OD)
    if 'gate_val' in activations:
        gate_val = activations['gate_val'][0].cpu().numpy() # (N, D)
        avg_gate = gate_val.mean()
        masked_gate = gate_val[mask[0].cpu().numpy()].mean()
        unmasked_gate = gate_val[~mask[0].cpu().numpy()].mean()
        print("\n=== Influence of Surrounding OD (Gate Value) ===")
        print(f"Average Gate Value (0 to 1): {avg_gate:.4f}")
        print(f"Masked Nodes Gate Value: {masked_gate:.4f}")
        print(f"Unmasked Nodes Gate Value: {unmasked_gate:.4f}")
        
    # 3. Attention Head Analysis
    # The transformer layers have multihead attention. We can inspect distance_bias as a proxy for how distance affects heads.
    if hasattr(model, 'distance_bias'):
        dist_bias_weights = model.distance_bias.weight.detach().cpu().numpy() # (50, nhead)
        print("\n=== Attention Head Roles (Distance Bias) ===")
        print("Heads vary by how they bias attention based on distance.")
        for head in range(dist_bias_weights.shape[1]):
            head_bias = dist_bias_weights[:, head]
            # Simple description: correlation with distance bucket
            corr = np.corrcoef(np.arange(50), head_bias)[0, 1]
            print(f"Head {head}: Correlation with distance = {corr:.4f} (Avg bias: {head_bias.mean():.4f})")
    
if __name__ == '__main__':
    best_model_dir = os.path.join(SRC_DIR, '../best_model')
    ckpt_name = 'mae_cpc:v5-64epoch.pth'
    ckpt_path = os.path.join(best_model_dir, ckpt_name)
    if not os.path.exists(ckpt_path):
        # Fallback to first available mae model
        ckpts = [f for f in os.listdir(best_model_dir) if f.startswith('mae') and (f.endswith('.pth') or f.endswith('.pt'))]
        if ckpts:
            ckpt_path = os.path.join(best_model_dir, ckpts[0])
    
    print(f"Using checkpoint: {ckpt_path}")
    explain_model(ckpt_path)
