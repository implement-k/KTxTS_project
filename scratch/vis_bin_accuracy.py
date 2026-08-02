import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(current_dir, '../src')))
sys.path.insert(0, os.path.abspath(os.path.join(current_dir, '../src/mae-year')))
from models import ODMAE
from dataset import ODDataset
from evaluation.fixed_eval_utils import make_base_data, apply_merge_events

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
    return model

print("Loading dataset...")
dataset = ODDataset(year='2023')
base_data = make_base_data(dataset)
val_meta_dict = torch.load(os.path.join(current_dir, '../dataset/fixed_eval/fixed_val_meta_2023.pt'), map_location='cpu', weights_only=False)
val_meta = []
for city, tasks in val_meta_dict.items():
    for task_id, samples in tasks.items():
        val_meta.extend(samples)

model = load_model(os.path.join(current_dir, "../best_model/mae:hybrid-86epoch.pth"))
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = model.to(device)

np.random.seed(42)
sampled_indices = np.random.choice(len(val_meta), 100, replace=False)

all_y_true = []
all_y_pred = []

print(f"Running inference on {len(val_meta)} samples...")
for idx in sampled_indices:
    meta = val_meta[idx]
    
    sample = apply_merge_events(
        base_data,
        meta['mask_indices'],
        meta['merge_events'],
        hide_indices=base_data['test_indices'],
        imputation_values=None
    )
    
    x_static = sample['X_static'].float().unsqueeze(0).to(device)
    x_dist = sample['X_dist'].float().unsqueeze(0).to(device)
    mask_t = sample['mask'].unsqueeze(0).to(device)
    x_od_masked = sample['X_OD_masked'].float().unsqueeze(0).to(device)
    a_spatial = sample['A_spatial'].float().unsqueeze(0).to(device)
    active_node_mask = sample['active_node_mask'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred = model(x_static, x_od_masked, x_dist, a_spatial, mask_t, active_node_mask)
        
    y_true_flat = sample['y_OD_raw'][sample['mask']].cpu().numpy()
    y_pred_flat = pred[0][sample['mask']].cpu().numpy()
    
    # Exponentials since the inputs are log1p scaled
    y_pred_flat = np.expm1(y_pred_flat)
    
    all_y_true.extend(y_true_flat)
    all_y_pred.extend(y_pred_flat)

y_true = np.array(all_y_true)
y_pred = np.array(all_y_pred)

print(f"Total masked pairs evaluated: {len(y_true)}")

bins = [0, 10, 50, 100, 500, 1000, 5000, 10000, np.inf]
bin_labels = ["0~10", "10~50", "50~100", "100~500", "500~1k", "1k~5k", "5k~10k", "10k+"]

rmse_list = []
mape_list = []
counts = []

for i in range(len(bins)-1):
    low, high = bins[i], bins[i+1]
    bin_mask = (y_true >= low) & (y_true < high)
    
    yt = y_true[bin_mask]
    yp = y_pred[bin_mask]
    
    counts.append(len(yt))
    
    if len(yt) > 0:
        rmse = np.sqrt(np.mean((yt - yp)**2))
        mape = np.mean(np.abs(yt - yp) / (yt + 1e-6)) * 100 
    else:
        rmse = 0
        mape = 0
        
    rmse_list.append(rmse)
    mape_list.append(mape)

fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:blue'
ax1.set_xlabel('True Traffic Volume Bins')
ax1.set_ylabel('RMSE', color=color)
ax1.bar(bin_labels, rmse_list, color=color, alpha=0.6, label='RMSE')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(bottom=0)

ax2 = ax1.twinx()  
color = 'tab:red'
ax2.set_ylabel('MAPE (%)', color=color)  
ax2.plot(bin_labels, mape_list, color=color, marker='o', linewidth=2, label='MAPE')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(bottom=0, top=min(1000, max(mape_list)+10) if counts else 100)

plt.title('Accuracy by Traffic Volume Bin (2023)')
fig.tight_layout()  
plt.savefig(os.path.join(current_dir, 'vis_bin_accuracy.png'))
print(f"Saved vis_bin_accuracy.png")

with open(os.path.join(current_dir, 'vis_bin_accuracy_results.txt'), 'w') as f:
    f.write("Bin\tCount\tRMSE\tMAPE(%)\n")
    for b, c, r, m in zip(bin_labels, counts, rmse_list, mape_list):
        f.write(f"{b}\t{c}\t{r:.2f}\t{m:.2f}\n")
