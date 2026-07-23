import numpy as np
import torch
import math
import random
from collections import defaultdict

def cpc_score(y_true, y_pred):
    numerator   = 2 * np.sum(np.minimum(y_true, y_pred))
    denominator = np.sum(y_true) + np.sum(y_pred)
    if denominator == 0:
        return 0.0
    return numerator / denominator

def validate_mae(model, dataset, val_indices, criterion, device):
    """
    Task A: 순수 마스킹 평가
    """
    val_mask_np = np.zeros(dataset.num_nodes, dtype=bool)
    val_mask_np[val_indices] = True

    # 100% masking for val nodes
    X_static_masked = dataset.X_static.copy()
    X_static_masked[val_mask_np, :-1] = 0.0
    X_static_masked[val_mask_np, -1] = 1.0

    X_OD_masked = dataset.X_OD.copy()
    X_OD_masked[val_mask_np, :] = 0
    X_OD_masked[:, val_mask_np] = 0

    x_s = torch.tensor(X_static_masked, dtype=torch.float32, device=device).unsqueeze(0)
    x_d = torch.tensor(dataset.X_dist,  dtype=torch.float32, device=device).unsqueeze(0)
    x_o = torch.tensor(X_OD_masked, dtype=torch.float32, device=device).unsqueeze(0)
    y_o = torch.tensor(dataset.X_OD, dtype=torch.float32, device=device).unsqueeze(0)
    mask = torch.tensor(val_mask_np, dtype=torch.bool, device=device).unsqueeze(0)
    active_node_mask = torch.ones_like(mask, dtype=torch.bool, device=device)

    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.eval()

    with torch.no_grad():
        v_pred_scale, v_pred_raw = model(x_s, x_o, x_d, mask, active_node_mask=active_node_mask)

    m2d = mask.unsqueeze(1) | mask.unsqueeze(2)
    diag_mask = torch.eye(dataset.num_nodes, device=device, dtype=torch.bool).unsqueeze(0).expand(v_pred_scale.shape[0], -1, -1)
    
    is_weibull = criterion.__class__.__name__ == 'HybridWeibullODLoss'
    if is_weibull:
        v_loss = criterion(v_pred_scale, v_pred_raw, y_o, m2d, diag_mask, active_node_mask=active_node_mask).item()
        pred_lambda = v_pred_scale.cpu().numpy()
        gamma_diag = math.gamma(1 + 1 / criterion.k_diag)
        gamma_offdiag = math.gamma(1 + 1 / criterion.k_offdiag)
        
        p_real = np.zeros_like(pred_lambda)
        diag_np = np.eye(dataset.num_nodes, dtype=bool)
        for b in range(p_real.shape[0]):
            p_real[b, diag_np] = pred_lambda[b, diag_np] * gamma_diag
            p_real[b, ~diag_np] = pred_lambda[b, ~diag_np] * gamma_offdiag
            
        p_real = p_real[m2d.cpu().numpy()]
    else:
        v_loss = criterion(v_pred_raw, y_o, 1.0, m2d).item()
        p_real = np.maximum(torch.expm1(v_pred_raw[m2d]).cpu().numpy(), 0)
        
    y_real = torch.expm1(y_o[m2d]).cpu().numpy()

    rmse = np.sqrt(np.mean((y_real - p_real) ** 2))
    cpc = cpc_score(y_real, p_real)

    return v_loss, rmse, cpc

def validate_mae_merge(model, dataset, val_indices, criterion, device):
    """
    Task B: 병합 평가
    """
    if not hasattr(dataset, 'merge_cache') or len(dataset.merge_cache) == 0:
        return 0.0, 0.0, 0.0
        
    val_set = set(val_indices)
    pairs = [k for k in dataset.merge_cache.keys() if k[0] in val_set]
    
    if len(pairs) == 0:
        return 0.0, 0.0, 0.0
        
    pairs_by_a = defaultdict(list)
    for a, b in pairs:
        pairs_by_a[a].append(b)
        
    selected_pairs = []
    # 고정된 시드로 평가 안정성 확보
    rng = random.Random(42)
    for a, b_list in pairs_by_a.items():
        rng.shuffle(b_list)
        for b in b_list[:2]: # 2개씩만 병합
            selected_pairs.append((a, b))
            
    B = len(selected_pairs)
    N = dataset.num_nodes
    
    X_s = np.zeros((B, N, dataset.X_static.shape[1]), dtype=np.float32)
    X_d = np.zeros((B, N, N), dtype=np.float32)
    X_o = np.zeros((B, N, N), dtype=np.float32)
    Y_o = np.zeros((B, N, N), dtype=np.float32)
    Mask = np.zeros((B, N), dtype=bool)
    ActiveMask = np.ones((B, N), dtype=bool)
    
    for i, (idx_a, idx_b) in enumerate(selected_pairs):
        y_OD = dataset.X_OD.copy()
        X_OD_masked = dataset.X_OD.copy()
        X_static_masked = dataset.X_static.copy()
        X_dist_curr = dataset.X_dist.copy()
        mask = np.zeros(N, dtype=bool)
        active_mask = np.ones(N, dtype=bool)
        
        mask[idx_a] = True
        
        # OD Masking for Task 1 part
        X_OD_masked[idx_a, :] = 0
        X_OD_masked[:, idx_a] = 0
        
        # Merge Cache
        cache = dataset.merge_cache[(idx_a, idx_b)]
        
        # OD Merge
        raw_od_a = np.expm1(y_OD[idx_a, :])
        raw_od_b = np.expm1(y_OD[idx_b, :])
        new_self_loop = (
            np.expm1(y_OD[idx_a, idx_a]) + np.expm1(y_OD[idx_b, idx_b]) +
            np.expm1(y_OD[idx_a, idx_b]) + np.expm1(y_OD[idx_b, idx_a])
        )
        
        raw_y_od_row_a = raw_od_a + raw_od_b
        raw_y_od_col_a = np.expm1(y_OD[:, idx_a]) + np.expm1(y_OD[:, idx_b])
        
        y_OD[idx_a, :] = np.log1p(raw_y_od_row_a)
        y_OD[:, idx_a] = np.log1p(raw_y_od_col_a)
        y_OD[idx_a, idx_a] = np.log1p(new_self_loop)
        
        # Distance Merge
        merged_dist = cache['merged_dist_row_at_a']
        X_dist_curr[idx_a, :] = np.log1p(merged_dist)
        X_dist_curr[:, idx_a] = np.log1p(merged_dist)
        
        # Static Merge
        merged_static_scaled = dataset.scaler.transform(cache['merged_raw_static_at_a'].reshape(1, -1))[0]
        # Masking it 100% since it's the target node
        X_static_masked[idx_a, :-1] = 0.0
        X_static_masked[idx_a, -1] = 1.0
        
        # Virtual Deletion
        active_mask[idx_b] = False
        X_static_masked[idx_b, :] = 0.0
        X_OD_masked[idx_b, :] = 0.0
        X_OD_masked[:, idx_b] = 0.0
        X_dist_curr[idx_b, :] = 5.5
        X_dist_curr[:, idx_b] = 5.5
        
        X_s[i] = X_static_masked
        X_d[i] = X_dist_curr
        X_o[i] = X_OD_masked
        Y_o[i] = y_OD
        Mask[i] = mask
        ActiveMask[i] = active_mask

    x_s = torch.tensor(X_s, dtype=torch.float32, device=device)
    x_d = torch.tensor(X_d, dtype=torch.float32, device=device)
    x_o = torch.tensor(X_o, dtype=torch.float32, device=device)
    y_o = torch.tensor(Y_o, dtype=torch.float32, device=device)
    mask = torch.tensor(Mask, dtype=torch.bool, device=device)
    active_node_mask = torch.tensor(ActiveMask, dtype=torch.bool, device=device)

    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.eval()

    with torch.no_grad():
        v_pred_scale, v_pred_raw = model(x_s, x_o, x_d, mask, active_node_mask=active_node_mask)

    m2d = mask.unsqueeze(1) | mask.unsqueeze(2)
    diag_mask = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0).expand(B, -1, -1)
    
    is_weibull = criterion.__class__.__name__ == 'HybridWeibullODLoss'
    if is_weibull:
        v_loss = criterion(v_pred_scale, v_pred_raw, y_o, m2d, diag_mask, active_node_mask=active_node_mask).item()
        pred_lambda = v_pred_scale.cpu().numpy()
        gamma_diag = math.gamma(1 + 1 / criterion.k_diag)
        gamma_offdiag = math.gamma(1 + 1 / criterion.k_offdiag)
        
        p_real_full = np.zeros_like(pred_lambda)
        diag_np = np.eye(N, dtype=bool)
        for b in range(B):
            p_real_full[b, diag_np] = pred_lambda[b, diag_np] * gamma_diag
            p_real_full[b, ~diag_np] = pred_lambda[b, ~diag_np] * gamma_offdiag
    else:
        v_loss = criterion(v_pred_raw, y_o, 1.0, m2d).item()
        p_real_full = np.maximum(torch.expm1(v_pred_raw).cpu().numpy(), 0)
        
    y_real_full = torch.expm1(y_o).cpu().numpy()

    # Active mask filter for CPC/RMSE calculation
    active_m2d = active_node_mask.unsqueeze(1) & active_node_mask.unsqueeze(2)
    valid_cells = (m2d & active_m2d).cpu().numpy()
    
    p_real_valid = p_real_full[valid_cells]
    y_real_valid = y_real_full[valid_cells]

    if len(y_real_valid) > 0:
        rmse = np.sqrt(np.mean((y_real_valid - p_real_valid) ** 2))
        cpc = cpc_score(y_real_valid, p_real_valid)
    else:
        rmse, cpc = 0.0, 0.0

    return v_loss, rmse, cpc