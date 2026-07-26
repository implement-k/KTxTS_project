import numpy as np
import torch


def apply_merge_events(base_data, mask_indices, merge_events):
    N = base_data['num_nodes']
    masking_indices = base_data['masking_indices']
    scaler = base_data['scaler']
    merge_cache = base_data['merge_cache']
    hide_indices = base_data['test_indices']  # val 평가 시 항상 숨겨야 하는 대상

    mask_indices_set = set(mask_indices)
    mask = np.zeros(N, dtype=bool)
    if len(mask_indices) > 0:
        mask[mask_indices] = True

    hide_mask = np.zeros(N, dtype=bool)
    if len(hide_indices) > 0:
        hide_mask[hide_indices] = True

    y_OD_raw = base_data['X_OD_raw'].copy()
    X_static_masked = base_data['X_static'].copy()          # 스케일된 버전(-2, -1 indicator 컬럼 포함)
    X_static_raw_masked = base_data['X_static_raw'].copy()
    X_dist_curr = base_data['X_dist_raw'].copy()
    active_node_mask = np.ones(N, dtype=bool)

    used_b_nodes = set()

    for idx_a, idx_b, event_type in merge_events:
        if idx_a in used_b_nodes or idx_b in used_b_nodes:
            continue

        if event_type == 'mask_with_known':
            primary_node, secondary_node = (idx_a, idx_b) if idx_b in mask_indices_set else (idx_b, idx_a)
        else:
            primary_node, secondary_node = (idx_a, idx_b)

        cache_key = (primary_node, secondary_node)
        if cache_key not in merge_cache:
            cache_key = (secondary_node, primary_node)
            if cache_key not in merge_cache:
                continue

        cache = merge_cache[cache_key]

        new_self_loop = (
            y_OD_raw[primary_node, primary_node] + y_OD_raw[secondary_node, secondary_node]
            + y_OD_raw[primary_node, secondary_node] + y_OD_raw[secondary_node, primary_node]
        )
        raw_row_a = y_OD_raw[primary_node, :] + y_OD_raw[secondary_node, :]
        raw_col_a = y_OD_raw[:, primary_node] + y_OD_raw[:, secondary_node]
        y_OD_raw[primary_node, :] = raw_row_a
        y_OD_raw[:, primary_node] = raw_col_a
        y_OD_raw[primary_node, primary_node] = new_self_loop

        active_node_mask[secondary_node] = False
        used_b_nodes.add(secondary_node)

        merged_raw_static = cache['merged_raw_static_at_a']
        merged_static = scaler.transform(merged_raw_static.reshape(1, -1))[0]

        merged_dist_row = cache['merged_dist_row_at_a']
        X_dist_curr[primary_node, :] = np.log1p(merged_dist_row)
        X_dist_curr[:, primary_node] = np.log1p(merged_dist_row)

        if event_type == 'known_merge':
            mask[primary_node] = False
            X_static_masked[primary_node, :-2] = merged_static
            X_static_raw_masked[primary_node, :] = merged_raw_static
            X_static_masked[primary_node, -2] = 0.0
            X_static_masked[primary_node, -1] = 0.0

        elif event_type == 'mask_with_mask':
            mask[primary_node] = True
            X_static_masked[primary_node, :-2] = merged_static
            X_static_raw_masked[primary_node, :] = merged_raw_static
            X_static_masked[primary_node, masking_indices] = 0.0
            X_static_raw_masked[primary_node, masking_indices] = 0.0
            X_static_masked[primary_node, -2] = 1.0
            X_static_masked[primary_node, -1] = 1.0

        elif event_type == 'mask_with_known':
            mask[primary_node] = False
            X_static_masked[primary_node, :-2] = merged_static
            X_static_raw_masked[primary_node, :] = merged_raw_static
            X_static_masked[primary_node, -2] = 0.0
            X_static_masked[primary_node, -1] = 1.0

    if len(mask_indices) > 0:
        X_static_masked[np.ix_(mask_indices, masking_indices)] = 0.0
        X_static_raw_masked[np.ix_(mask_indices, masking_indices)] = 0.0

    base_mask = mask | hide_mask
    if np.any(hide_mask):
        X_static_masked[hide_mask, :-2] = 0.0
        X_static_raw_masked[hide_mask, :-2] = 0.0

    X_static_masked[base_mask, -2] = 1.0
    X_static_masked[base_mask, -1] = 0.0
    X_static_raw_masked[base_mask, -2] = 1.0
    X_static_raw_masked[base_mask, -1] = 0.0

    y_OD = np.log1p(y_OD_raw)
    X_OD_masked = y_OD.copy()
    final_mask = mask | hide_mask
    X_OD_masked[final_mask, :] = 0.0
    X_OD_masked[:, final_mask] = 0.0
    for b in used_b_nodes:
        X_OD_masked[b, :] = 0.0
        X_OD_masked[:, b] = 0.0

    inactive = ~active_node_mask
    X_dist_curr[inactive, :] = 5.5
    X_dist_curr[:, inactive] = 5.5
    X_dist_curr = np.where(np.isnan(X_dist_curr), 5.5, X_dist_curr)

    return {
        'X_static': torch.tensor(X_static_masked, dtype=torch.float16),
        'X_static_raw': torch.tensor(X_static_raw_masked, dtype=torch.float32),
        'X_dist': torch.tensor(X_dist_curr, dtype=torch.float16),
        'X_OD_masked': torch.tensor(X_OD_masked, dtype=torch.float16),
        'y_OD': torch.tensor(y_OD, dtype=torch.float16),
        'y_OD_raw': torch.tensor(y_OD_raw, dtype=torch.float32),
        'mask': torch.tensor(mask, dtype=torch.bool),
        'active_node_mask': torch.tensor(active_node_mask, dtype=torch.bool),
        'loss_mask': torch.tensor(mask.copy(), dtype=torch.bool),
    }


def make_base_data(dataset):
    """ODDataset 인스턴스에서 연도당 1번만 저장할 원본 정보를 추출."""
    return {
        'num_nodes': dataset.num_nodes,
        'masking_indices': dataset.masking_indices,
        'scaler': dataset.scaler,
        'merge_cache': dataset.merge_cache,
        'test_indices': dataset.test_indices,
        'X_static': dataset.X_static,          # 스케일 + indicator 컬럼까지 포함된 원본(마스킹 전)
        'X_static_raw': dataset.X_static_raw,
        'X_dist_raw': dataset.X_dist_raw,
        'X_OD_raw': dataset.X_OD_raw,
    }
    
    
# from fixed_eval_utils import apply_merge_events

# base_data = torch.load('dataset/fixed_eval/base_data_2023.pt')
# val_meta = torch.load('dataset/fixed_eval/fixed_val_meta_2023.pt')

# for task in [0, 1, 2, 3, 4]:
#     for meta in val_meta['동탄'][task]:
#         sample = apply_merge_events(base_data, meta['mask_indices'], meta['merge_events'])
#         # sample['X_static'], sample['y_OD'] 등을 모델에 그대로 넣으면 됨