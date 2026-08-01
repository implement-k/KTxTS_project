import os
import sys
import torch
import numpy as np

sys.path.insert(0, './src')
sys.path.insert(0, './src/mae-year')

from models import ODMAE
from dataset import ODDataset
from evaluation.fixed_eval_utils import make_base_data, apply_merge_events

def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    sum_both = np.sum(y_true + y_pred)
    cpc = 2 * np.sum(np.minimum(y_true, y_pred)) / sum_both if sum_both > 0 else 0
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    prmse = rmse / np.mean(y_true) if np.mean(y_true) > 0 else 0
    return {'cpc': cpc, 'rmse': rmse, 'prmse': prmse}

def format_metrics(m):
    return f"CPC={m['cpc']:.4f}, RMSE={m['rmse']:.4f}, %RMSE={m['prmse']:.4f}"

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
    
    # Add dummy hook to prevent PyTorch in-place NaNs in loop
    def check_nan_hook(module, inp, out):
        pass
    for name, module in model.named_modules():
        module.register_forward_hook(check_nan_hook)
        
    return model

print("Loading dataset...")
dataset = ODDataset(year='2023')
base_data = make_base_data(dataset)
val_meta = torch.load('dataset/fixed_eval/fixed_val_meta_2023.pt', map_location='cpu', weights_only=False)

print("Loading models...")
model1 = load_model("best_model/mae:v5-14epoch.pth")
model2 = load_model("best_model/mae_cpc:v5-64epoch.pth")

print("Evaluating Ensemble...")

y_true_list = []
y_pred_avg_list = []
y_pred_w1_list = [] # 0.3 mae, 0.7 mae_cpc
y_pred_w2_list = [] # 0.7 mae, 0.3 mae_cpc

total_samples = 0
for city, tasks in val_meta.items():
    for task_id, sample_metas in tasks.items():
        for meta in sample_metas:
            sample = apply_merge_events(base_data, meta['mask_indices'], meta['merge_events'], hide_indices=base_data['test_indices'])
            
            x_static = sample['X_static'].float().unsqueeze(0)
            x_dist = sample['X_dist'].float().unsqueeze(0)
            mask_t = sample['mask'].unsqueeze(0)
            x_od_masked = sample['X_OD_masked'].float().unsqueeze(0)
            a_spatial = sample['A_spatial'].float().unsqueeze(0)
            active_node_mask = sample['active_node_mask'].unsqueeze(0)
            
            with torch.no_grad():
                pred1 = model1(x_static, x_od_masked, x_dist, a_spatial, mask_t, active_node_mask)
                pred2 = model2(x_static, x_od_masked, x_dist, a_spatial, mask_t, active_node_mask)
                
            # pred[0] is od_pred_final. pred1 is tuple or tensor depending on use_self_loop_predictor
            out1 = pred1[0] if isinstance(pred1, tuple) else pred1
            out2 = pred2[0] if isinstance(pred2, tuple) else pred2
            
            T_pred1 = torch.expm1(out1).cpu().numpy().squeeze(0)
            T_pred2 = torch.expm1(out2).cpu().numpy().squeeze(0)
            
            T_pred1 = np.maximum(T_pred1, 0)
            T_pred2 = np.maximum(T_pred2, 0)
            
            T_pred_avg = (T_pred1 + T_pred2) / 2.0
            T_pred_w1 = 0.3 * T_pred1 + 0.7 * T_pred2
            T_pred_w2 = 0.7 * T_pred1 + 0.3 * T_pred2
            
            y_od = sample['y_OD_raw'].cpu().numpy()
            
            # Evaluate only on mask_indices interacting with active nodes
            eval_indices = np.array(meta['mask_indices'])
            N = y_od.shape[0]
            
            eval_mask_2d = np.zeros((N, N), dtype=bool)
            eval_mask_2d[:, eval_indices] = True
            eval_mask_2d[eval_indices, :] = True
            
            active_m2d = active_node_mask[0].cpu().numpy().reshape(-1, 1) & active_node_mask[0].cpu().numpy().reshape(1, -1)
            valid_cells = eval_mask_2d & active_m2d
            
            y_true_list.append(y_od[valid_cells])
            y_pred_avg_list.append(T_pred_avg[valid_cells])
            y_pred_w1_list.append(T_pred_w1[valid_cells])
            y_pred_w2_list.append(T_pred_w2[valid_cells])
            
            total_samples += 1

print(f"Evaluated {total_samples} samples.")

y_true_all = np.concatenate(y_true_list)
y_pred_avg_all = np.concatenate(y_pred_avg_list)
y_pred_w1_all = np.concatenate(y_pred_w1_list)
y_pred_w2_all = np.concatenate(y_pred_w2_list)

print("\\n[1] 단순평균 (Simple Average) Metrics:")
res_avg = compute_metrics(y_true_all, y_pred_avg_all)
print(format_metrics(res_avg))

print("\\n[2] 가중평균 (0.3 mae:v5 + 0.7 mae_cpc:v5) Metrics:")
res_w1 = compute_metrics(y_true_all, y_pred_w1_all)
print(format_metrics(res_w1))

print("\\n[3] 가중평균 (0.7 mae:v5 + 0.3 mae_cpc:v5) Metrics:")
res_w2 = compute_metrics(y_true_all, y_pred_w2_all)
print(format_metrics(res_w2))
