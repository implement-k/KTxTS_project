import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, './src')
sys.path.insert(0, './src/mae-year')
from models import ODMAE
from dataset import ODDataset
from evaluation.fixed_eval_utils import make_base_data, apply_merge_events

def load_model(path, d_model=128):
    st = torch.load(path, map_location='cpu')
    d_model = st['feature_embed.0.weight'].shape[0]
    num_features = st['feature_embed.0.weight'].shape[1]
    has_mask = 'mask_proj.weight' in st
    nhead = st['distance_bias.weight'].shape[1] if 'distance_bias.weight' in st else 4
    has_self_loop = 'self_loop_predictor.0.weight' in st
    
    model = ODMAE(num_features=num_features, d_model=d_model, nhead=nhead, num_layers=4,
                  od_embed_layers=2, use_distance_friction=True, use_self_loop_predictor=has_self_loop,
                  use_mask_channel=has_mask)
    model.load_state_dict(st, strict=False)
    model.eval()
    
    def check_nan_hook(module, inp, out):
        if isinstance(out, torch.Tensor):
            if out.isnan().any():
                print(f"NaN found in output of {module.__class__.__name__}")
        elif isinstance(out, tuple):
            for i, o in enumerate(out):
                if isinstance(o, torch.Tensor) and o.isnan().any():
                    print(f"NaN found in output of {module.__class__.__name__} at tuple index {i}")

    for name, module in model.named_modules():
        module.register_forward_hook(check_nan_hook)
        
    return model

print("Loading dataset...")
dataset = ODDataset(year='2023')
base_data = make_base_data(dataset)
val_meta = torch.load('dataset/fixed_eval/fixed_val_meta_2023.pt', map_location='cpu', weights_only=False)

models = {
    "mae_cpc:v5": load_model("best_model/mae_cpc:v5.pth"),
    "mae:v5": load_model("best_model/mae:v5.pth")
}

for name, model in models.items():
    print(f"\n=====================================")
    print(f"Model: {name}")
    print(f"=====================================")
    
    if hasattr(model, 'mask_token_high'):
        print("mask_token_high norm:", model.mask_token_high.norm().item())
        print("mask_token_low norm:", model.mask_token_low.norm().item())
    
    task = 0
    meta = val_meta['동탄'][task][0] # just pick the first sample
    sample = apply_merge_events(base_data, meta['mask_indices'], meta['merge_events'], hide_indices=base_data['test_indices'])
    
    x_static = sample['X_static'].float().unsqueeze(0)
    x_dist = sample['X_dist'].float().unsqueeze(0)
    mask_t = sample['mask'].unsqueeze(0)
    x_od_masked = sample['X_OD_masked'].float().unsqueeze(0)
    a_spatial = sample['A_spatial'].float().unsqueeze(0)
    active_node_mask = sample['active_node_mask'].unsqueeze(0)
    
    with torch.no_grad():
        pred = model(x_static, x_od_masked, x_dist, a_spatial, mask_t, active_node_mask)
        
    T_pred = torch.expm1(pred[0]).cpu().numpy()
    y_od = sample['y_OD_raw'].cpu().numpy()
    
    eval_indices = np.array(meta['mask_indices'])
    N = y_od.shape[0]
    
    eval_mask_2d = np.zeros((N, N), dtype=bool)
    eval_mask_2d[:, eval_indices] = True
    eval_mask_2d[eval_indices, :] = True
    
    active_m2d = active_node_mask[0].cpu().numpy().reshape(-1, 1) & active_node_mask[0].cpu().numpy().reshape(1, -1)
    valid_cells = eval_mask_2d & active_m2d
    
    y_od_eval = y_od[valid_cells]
    y_pred_eval = np.maximum(T_pred[valid_cells], 0)
    
    print("T_pred 상위 20개:", np.sort(y_pred_eval)[-20:])
    print("y_true 상위 20개:", np.sort(y_od_eval)[-20:])
    
    O_pred = np.maximum(T_pred, 0).sum(axis=1)
    O_true = y_od.sum(axis=1)
    
    valid_nodes = active_node_mask[0].cpu().numpy()
    O_pred_valid = O_pred[valid_nodes]
    O_true_valid = O_true[valid_nodes]
    
    print("O_pred 상위 20개:", np.sort(O_pred_valid)[-20:])
    print("O_true 상위 20개:", np.sort(O_true_valid)[-20:])
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(y_od_eval, y_pred_eval, alpha=0.5, s=2)
    plt.plot([0, y_od_eval.max()], [0, y_od_eval.max()], 'r--')
    plt.xlabel('y_true')
    plt.ylabel('T_pred')
    plt.title(f'{name} - T_pred vs y_true')
    
    plt.subplot(1, 2, 2)
    plt.scatter(O_true_valid, O_pred_valid, alpha=0.5, s=2)
    plt.plot([0, O_true_valid.max()], [0, O_true_valid.max()], 'r--')
    plt.xlabel('O_true')
    plt.ylabel('O_pred')
    plt.title(f'{name} - O_pred vs O_true')
    
    plt.tight_layout()
    img_path = os.path.abspath(f'scratch/vis_{name.replace(":", "_")}.png')
    plt.savefig(img_path)
    print(f"Saved {img_path}")
