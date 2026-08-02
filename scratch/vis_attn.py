import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, './src')
sys.path.insert(0, './src/mae-year')
from models import ODMAE
from dataset import ODDataset
from evaluation.fixed_eval_utils import make_base_data, apply_merge_events
import matplotlib.font_manager as fm

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

def get_dong_names():
    df = pd.read_excel('dataset/raw/dong/OD_dong_list_2023.xlsx')
    idx2name = {}
    for idx, row in df.iterrows():
        idx2name[idx] = row['dong_name'] if 'dong_name' in row else str(row['dong_code'])
    return idx2name

st = torch.load('best_model/mae_cpc:v5-64epoch.pth', map_location='cpu')
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

# Dummy hook to fix in-place NaNs
def check_nan_hook(module, inp, out): pass
for name, module in model.named_modules():
    module.register_forward_hook(check_nan_hook)

dataset = ODDataset(year='2023')
base_data = make_base_data(dataset)
idx2name = get_dong_names()

val_meta = torch.load('dataset/fixed_eval/fixed_val_meta_2023.pt', map_location='cpu', weights_only=False)
meta = val_meta['동탄'][0][0]
sample = apply_merge_events(base_data, meta['mask_indices'], meta['merge_events'], hide_indices=base_data['test_indices'])

x_static = sample['X_static'].float().unsqueeze(0)
x_dist = sample['X_dist'].float().unsqueeze(0)
mask_t = sample['mask'].unsqueeze(0)
x_od_masked = sample['X_OD_masked'].float().unsqueeze(0)
a_spatial = sample['A_spatial'].float().unsqueeze(0)
active_node_mask = sample['active_node_mask'].unsqueeze(0)

# Hook to capture transformer input
saved_args = {}
def pre_hook(module, args, kwargs):
    saved_args['x'] = args[0]
    saved_args['mask'] = kwargs.get('mask')
model.transformer.register_forward_pre_hook(pre_hook, with_kwargs=True)

with torch.no_grad():
    _ = model(x_static, x_od_masked, x_dist, a_spatial, mask_t, active_node_mask)

x_in = saved_args['x']
mask_in = saved_args['mask']

attn_output, attn_weights = model.transformer.layers[0].self_attn(
    x_in, x_in, x_in, attn_mask=mask_in, need_weights=True, average_attn_weights=False
)

# attn_weights shape: (B, nhead, N, N) -> (1, 4, N, N)
attn = attn_weights[0].detach().cpu().numpy()  # (4, N, N)
nhead, N, _ = attn.shape

dongtan_nodes = [1050, 1051, 1052]

for node_idx in dongtan_nodes:
    node_name = idx2name.get(node_idx, str(node_idx))
    
    fig, axes = plt.subplots(1, nhead, figsize=(nhead * 5, 5))
    fig.suptitle(f"Attention Weights for {node_name} (Node {node_idx}) in First Layer", fontsize=16)
    
    for h in range(nhead):
        # Attention from node_idx to all other nodes
        weights = attn[h, node_idx, :]
        
        # Get top 10 attended nodes
        top_indices = np.argsort(weights)[::-1][:10]
        top_weights = weights[top_indices]
        top_names = [idx2name.get(i, str(i)) for i in top_indices]
        
        ax = axes[h]
        sns.barplot(x=top_weights, y=top_names, ax=ax, palette="viridis")
        ax.set_title(f"Head {h+1}")
        ax.set_xlabel("Attention Weight")
        
    plt.tight_layout()
    plt.savefig(f"scratch/attn_{node_idx}.png")
    plt.close()
    
print("Saved attention plots to scratch/attn_*.png")
