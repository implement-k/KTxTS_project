import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(current_dir, '../src/mae-year')))
sys.path.insert(0, os.path.abspath(os.path.join(current_dir, '../src/gravity(경훈)')))
from dataset import ODDataset as GravityDataset
from model import DoublyConstrainedGravityModel
from evaluation.fixed_eval_utils import make_base_data, apply_merge_events

print("Loading dataset...")
datasets = []
X_static_train_list, X_o_train_list, X_d_train_list = [], [], []

for year in ['2019', '2023']:
    ds = GravityDataset(year=year, imputation='mean', use_raw_static=True)
    X_static_train_list.append(ds.X_static_train)
    X_o_train_list.append(ds.y_o[ds.train_mask])
    X_d_train_list.append(ds.y_d[ds.train_mask])
    datasets.append(ds)

X_static_train = np.concatenate(X_static_train_list, axis=0)
X_o_train = np.concatenate(X_o_train_list, axis=0)
X_d_train = np.concatenate(X_d_train_list, axis=0)

model = DoublyConstrainedGravityModel(generation_model_type='cross_class', beta=2.0, max_iter=10)
print("Training Gravity Model (cross_class)...")
model.fit_O_D(X_static_train, X_o_train, X_d_train, useLog=True)

dataset_2023 = GravityDataset(year='2023', imputation='mean', use_raw_static=True)
base_data = torch.load(os.path.join(current_dir, '../dataset/fixed_eval/base_data_2023.pt'), map_location='cpu', weights_only=False)
if 'X_dist' not in base_data and 'X_dist_raw' in base_data:
    base_data['X_dist'] = np.log1p(base_data['X_dist_raw'])
if 'A_spatial' not in base_data:
    # Gravity doesn't use A_spatial, but apply_merge_events expects it
    base_data['A_spatial'] = np.zeros((base_data['num_nodes'], base_data['num_nodes']), dtype=np.float32)
val_meta_dict = torch.load(os.path.join(current_dir, '../dataset/fixed_eval/fixed_val_meta_2023.pt'), map_location='cpu', weights_only=False)

val_meta = []
for city, tasks in val_meta_dict.items():
    for task_id, samples in tasks.items():
        val_meta.extend(samples)

np.random.seed(42)
sampled_indices = np.random.choice(len(val_meta), 100, replace=False)

all_y_true = []
all_y_pred = []

print(f"Running inference on {len(sampled_indices)} samples for Bin Accuracy...")
for idx in sampled_indices:
    meta = val_meta[idx]
    
    sample = apply_merge_events(
        base_data,
        meta['mask_indices'],
        meta['merge_events'],
        hide_indices=base_data['test_indices'],
        imputation_values=None
    )
    
    # Gravity 모델은 X_static_raw 사용
    O_pred, D_pred = model.predict_O_D(sample['X_static_raw'].float().numpy(), useLog=True)
    
    dist_matrix = sample['X_dist'].float().numpy()
    dist_no_diag = dist_matrix.copy()
    np.fill_diagonal(dist_no_diag, np.inf)
    intrazonal_dist = dist_no_diag.min(axis=1) / 2.0
    dist_matrix_modified = dist_matrix.copy()
    np.fill_diagonal(dist_matrix_modified, intrazonal_dist)
    
    T_pred = model.apply_ipf(O_pred, D_pred, dist_matrix_modified)
    
    mask = sample['mask'].numpy()
    y_true_flat = sample['y_OD_raw'].numpy()[mask]
    y_pred_flat = T_pred[mask]
    
    all_y_true.extend(y_true_flat)
    all_y_pred.extend(y_pred_flat)

all_y_true = np.array(all_y_true)
all_y_pred = np.array(all_y_pred)

print("Plotting Bin Accuracy...")
bins = [0, 10, 50, 100, 500, 1000, 10000, np.inf]
bin_labels = ['0-10', '10-50', '50-100', '100-500', '500-1k', '1k-10k', '10k+']
bin_indices = np.digitize(all_y_true, bins)

rmses = []
mapes = []

for i in range(1, len(bins)):
    mask = (bin_indices == i)
    if not np.any(mask):
        rmses.append(0)
        mapes.append(0)
        continue
    y_t = all_y_true[mask]
    y_p = all_y_pred[mask]
    rmse = np.sqrt(np.mean((y_t - y_p) ** 2))
    mape = np.mean(np.abs(y_t - y_p) / (y_t + 1e-6)) * 100
    rmses.append(rmse)
    mapes.append(mape)

fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:red'
ax1.set_xlabel('True Volume Bin')
ax1.set_ylabel('RMSE', color=color)
ax1.plot(bin_labels, rmses, marker='o', color=color, label='RMSE')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('MAPE (%)', color=color)
ax2.plot(bin_labels, mapes, marker='s', color=color, label='MAPE')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Performance by True Volume Bin (Gravity: CrossClassification)')
fig.tight_layout()
plt.grid(True, alpha=0.3)
img_path = os.path.join(current_dir, 'vis_cc_gravity_bin_accuracy.png')
plt.savefig(img_path)
print(f"Saved {img_path}")

# Node specific visualization
print("Running Node Specific Visualization...")
for task_id in [0]:
    for meta in val_meta_dict['동탄'][task_id][:1]:
        sample = apply_merge_events(
            base_data,
            meta['mask_indices'],
            meta['merge_events'],
            hide_indices=base_data['test_indices'],
            imputation_values=None
        )
        
        O_pred, D_pred = model.predict_O_D(sample['X_static_raw'].float().numpy(), useLog=True)
        dist_matrix = sample['X_dist'].float().numpy()
        dist_no_diag = dist_matrix.copy()
        np.fill_diagonal(dist_no_diag, np.inf)
        intrazonal_dist = dist_no_diag.min(axis=1) / 2.0
        dist_matrix_modified = dist_matrix.copy()
        np.fill_diagonal(dist_matrix_modified, intrazonal_dist)
        
        T_pred = model.apply_ipf(O_pred, D_pred, dist_matrix_modified)
        
        masked_nodes = meta['mask_indices']
        for node_idx in masked_nodes[:3]:
            y_true_row = sample['y_OD_raw'].numpy()[node_idx]
            y_pred_row = T_pred[node_idx]
            
            mask_row = sample['mask'].numpy()[node_idx]
            masked_indices = np.where(mask_row)[0]
            
            # 0인 타겟 필터링 (mae:hybrid 시각화와 동일한 기준 적용)
            valid_indices = [idx for idx in masked_indices if y_true_row[idx] > 0]
            if len(valid_indices) == 0:
                print(f"Node {node_idx} has no valid targets.")
                continue
                
            valid_indices = np.array(valid_indices)
            sorted_indices = valid_indices[np.argsort(y_true_row[valid_indices])]
            
            # 값이 너무 많으면 상위 20개만 시각화 (mae:hybrid 시각화와 동일)
            if len(sorted_indices) > 20:
                sorted_indices = sorted_indices[-20:]
                
            sorted_y_true = y_true_row[sorted_indices]
            sorted_y_pred = y_pred_row[sorted_indices]
            
            plt.figure(figsize=(10, 5))
            plt.plot(sorted_y_true, label='True OD', marker='o', alpha=0.7)
            plt.plot(sorted_y_pred, label='Pred OD (Gravity: CrossClass)', marker='x', alpha=0.7)
            plt.title(f'Node {node_idx} OD Predictions (Gravity: CrossClass)')
            plt.xlabel('Destination Node (Sorted by True Volume)')
            plt.ylabel('Traffic Volume')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            node_img_path = os.path.join(current_dir, f'vis_cc_gravity_node_{node_idx}.png')
            plt.savefig(node_img_path)
            print(f"Saved {node_img_path}")

print("Done.")
