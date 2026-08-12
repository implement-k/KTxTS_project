import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, '../src'))
sys.path.insert(0, os.path.join(current_dir, '../src/mae-year'))
from models import ODMAE
from dataset import ODDataset
from evaluation.fixed_eval_utils import make_base_data, apply_merge_events

# Set font for Korean text
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

def load_model(path, d_model=128):
    st = torch.load(path, map_location='cpu')
    if 'model_state_dict' in st:
        st = st['model_state_dict']
    d_model = st['feature_embed.0.weight'].shape[0]
    num_features = st['feature_embed.0.weight'].shape[1]
    has_mask = 'mask_proj.weight' in st
    nhead = st['distance_bias.weight'].shape[1] if 'distance_bias.weight' in st else 4
    has_self_loop = 'self_loop_predictor.0.weight' in st
    
    model = ODMAE(num_features=num_features, d_model=d_model, nhead=nhead, num_layers=4,
                  od_embed_layers=2, use_distance_friction=False, use_self_loop_predictor=has_self_loop,
                  use_mask_channel=has_mask)
    model.load_state_dict(st, strict=False)
    model.eval()
    
    def check_nan_hook(module, inp, out):
        pass # The mere presence of this hook prevents in-place NaN bugs in PyTorch evaluating x_dist/A_spatial
        
    for name, module in model.named_modules():
        module.register_forward_hook(check_nan_hook)
        
    return model

print("Loading dataset...")
dataset = ODDataset(year='2023')
base_data = make_base_data(dataset)
val_meta = torch.load(os.path.join(current_dir, '../dataset/fixed_eval/fixed_val_meta_2023.pt'), map_location='cpu', weights_only=False)

models = {
    "mae:hybrid": load_model(os.path.join(current_dir, "../best_model/mae_best.pth"))
}

city_name = '동탄'
task = 0
meta = val_meta[city_name][task][0] 

# Apply merge events
sample = apply_merge_events(base_data, meta['mask_indices'], meta['merge_events'], hide_indices=base_data['test_indices'])

x_static = sample['X_static'].float().unsqueeze(0)
x_dist = sample['X_dist'].float().unsqueeze(0)
mask_t = sample['mask'].unsqueeze(0)
x_od_masked = sample['X_OD_masked'].float().unsqueeze(0)
a_spatial = sample['A_spatial'].float().unsqueeze(0)
active_node_mask = sample['active_node_mask'].unsqueeze(0)

masked_nodes = meta['mask_indices']

for i, node_idx in enumerate(masked_nodes[:3]):
    plt.figure(figsize=(15, 5))
    
    for j, (name, model) in enumerate(models.items()):
        with torch.no_grad():
            pred = model(x_static, x_od_masked, x_dist, a_spatial, mask_t, active_node_mask)
            
        T_pred = torch.expm1(pred[0]).cpu().numpy()
        T_pred = np.maximum(T_pred, 0)
        y_od = sample['y_OD_raw'].cpu().numpy()
        
        row_true = y_od[node_idx, :]
        col_true = y_od[:, node_idx]
        
        row_pred = T_pred[node_idx, :]
        col_pred = T_pred[:, node_idx]
        
        combined_true = np.concatenate([row_true, col_true])
        combined_pred = np.concatenate([row_pred, col_pred])
        
        active_1d = active_node_mask[0].cpu().numpy()
        active_combined = np.concatenate([active_1d, active_1d])
        
        valid_mask = active_combined & (combined_true > 0)
        c_true_valid = combined_true[valid_mask]
        c_pred_valid = combined_pred[valid_mask]
        
        if len(c_true_valid) == 0: continue
        top20_indices = np.argsort(c_true_valid)[-20:]
        
        top20_true = c_true_valid[top20_indices]
        top20_pred = c_pred_valid[top20_indices]
        
        print(f"--- Node {node_idx} Model {name} ---")
        print(f"TRUE: {top20_true}")
        print(f"PRED: {top20_pred}")
        
        plt.subplot(1, 2, j+1)
        plt.plot(range(len(top20_true)), top20_true, 'o-', color='blue', label='Ground Truth')
        plt.plot(range(len(top20_pred)), top20_pred, 's-', color='red', label='Prediction')
        plt.title(f'Node {node_idx} Top 20 OD - {name}')
        plt.xlabel('OD Pair (Sorted by True Volume)')
        plt.ylabel('Traffic Volume')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
    plt.tight_layout()
    img_path = os.path.join(current_dir, f'vis_node_{node_idx}.png')
    plt.savefig(img_path)
    print(f"Saved {img_path}")
