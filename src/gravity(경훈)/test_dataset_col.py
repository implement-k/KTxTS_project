import torch, os, sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'evaluation'))
from fixed_eval_utils import apply_merge_events

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
base_data = torch.load(os.path.join(base_dir, 'dataset', 'fixed_eval', 'base_data_2023.pt'), weights_only=False)
val_meta = torch.load(os.path.join(base_dir, 'dataset', 'fixed_eval', 'fixed_val_meta_2023.pt'), weights_only=False)

# 1. shape과 dtype 확인
sample = apply_merge_events(base_data, val_meta['동탄'][1][0]['mask_indices'], val_meta['동탄'][1][0]['merge_events'])
print("X_static_raw shape:", sample['X_static_raw'].shape)
print("X_static_raw dtype:", sample['X_static_raw'].dtype)
print("컬럼 개수:", sample['X_static_raw'].shape[1])

# 2. 실제 컬럼명이 있는지 base_data / dataset에서 확인
print("\nbase_data keys:", base_data.keys())
if 'feature_cols' in base_data:
    print("feature_cols:", base_data['feature_cols'])
elif 'masking_indices' in base_data:
    print("masking_indices (컬럼 인덱스만):", base_data['masking_indices'])