import numpy as np
import torch


def _coerce_numeric_raw_static(raw_static, expected_len):
    """merge_cache에 코드/지역명 같은 메타 컬럼이 섞여 있으면 숫자 feature만 추출한다.

    일부 2023 merge_cache는 [코드, '서울_송파구', feature...]처럼 저장되어 있어
    gravity baseline의 LGBM 입력으로 바로 넣으면 문자열 float 변환 오류가 난다.
    fixed_eval을 다시 만들지 않고 평가할 수 있도록 숫자 feature만 남긴다.
    """
    values = np.asarray(raw_static, dtype=object).reshape(-1)

    if values.size == expected_len:
        return values.astype(np.float32)

    numeric_values = []
    for value in values:
        try:
            numeric_values.append(float(value))
        except (TypeError, ValueError):
            continue

    if len(numeric_values) >= expected_len:
        return np.asarray(numeric_values[-expected_len:], dtype=np.float32)

    raise ValueError(
        f"merged_raw_static에서 숫자 feature {expected_len}개를 만들 수 없음 "
        f"(raw_len={values.size}, numeric_len={len(numeric_values)})"
    )


def apply_merge_events(base_data, mask_indices, merge_events, hide_indices=None, imputation_values=None):
    """
    fixed_eval 샘플을 실제 평가 입력 형태로 복원한다.

    mask_indices:
        이번 val/test 샘플에서 예측해야 하는 대상 동.
        이 동들은 masking_indices에 해당하는 4개 feature
        (worker_count, business_count, worker_density, business_density)만 0으로 마스킹한다.

    hide_indices:
        평가 대상은 아니지만 입력에서 완전히 숨겨야 하는 holdout 동.
        validation에서는 test 동을 보면 치팅이므로 base_data['test_indices']를 넘긴다.
        test에서는 test 동 자체가 평가 대상이므로 []를 넘겨서 전체 feature를 숨기지 않는다.
        None은 기존 호출과의 호환을 위해 validation 방식(base_data['test_indices'])으로 처리한다.
    """
    N = base_data['num_nodes']
    masking_indices = base_data['masking_indices']
    scaler = base_data['scaler']
    merge_cache = base_data['merge_cache']
    # Backward-compatible default: 기존 코드처럼 validation holdout(test 동)을 숨긴다.
    # train_and_val.py에서는 split_name에 따라 val/test를 명시적으로 구분해서 넘긴다.
    if hide_indices is None:
        hide_indices = base_data['test_indices']

    # imputation_values가 없으면 기존 zero-imputation과 동일하게 동작한다.
    # mean-imputation 실험에서는 train_and_val.py가 train 동 평균을 넘겨주고,
    # 아래 mask_indices 4개 feature만 그 평균값으로 채운다.
    if imputation_values is None:
        scaled_impute_values = 0.0
        raw_impute_values = 0.0
    else:
        scaled_impute_values = np.asarray(imputation_values['scaled'], dtype=np.float32)[masking_indices]
        raw_impute_values = np.asarray(imputation_values['raw'], dtype=np.float32)[masking_indices]

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
    X_dist_curr_raw = base_data['X_dist_raw'].copy()
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

        # X_static은 항상 indicator 2개가 끝에 붙어있음.
        F = base_data['X_static'].shape[1] - 2
        raw_has_indicators = (base_data['X_static_raw'].shape[1] == base_data['X_static'].shape[1])
        
        merged_raw_static = _coerce_numeric_raw_static(
            cache['merged_raw_static_at_a'],
            F,
        )
        merged_static = scaler.transform(merged_raw_static.reshape(1, -1))[0]
        print(f"merged_static range: min={merged_static.min():.2f} max={merged_static.max():.2f}")

        merged_dist_row = cache['merged_dist_row_at_a']
        X_dist_curr[primary_node, :] = np.log1p(merged_dist_row)
        X_dist_curr[:, primary_node] = np.log1p(merged_dist_row)
        X_dist_curr_raw[primary_node, :] = merged_dist_row          # ← 추가: raw 그대로
        X_dist_curr_raw[:, primary_node] = merged_dist_row 

        if event_type == 'known_merge':
            mask[primary_node] = False
            X_static_masked[primary_node, :-2] = merged_static
            if raw_has_indicators:
                X_static_raw_masked[primary_node, :-2] = merged_raw_static
            else:
                X_static_raw_masked[primary_node, :] = merged_raw_static
            X_static_masked[primary_node, -2] = 0.0
            X_static_masked[primary_node, -1] = 0.0

        elif event_type == 'mask_with_mask':
            mask[primary_node] = True
            X_static_masked[primary_node, :-2] = merged_static
            if raw_has_indicators:
                X_static_raw_masked[primary_node, :-2] = merged_raw_static
            else:
                X_static_raw_masked[primary_node, :] = merged_raw_static
            X_static_masked[primary_node, masking_indices] = scaled_impute_values
            X_static_raw_masked[primary_node, masking_indices] = raw_impute_values
            X_static_masked[primary_node, -2] = 1.0
            X_static_masked[primary_node, -1] = 1.0

        elif event_type == 'mask_with_known':
            mask[primary_node] = False
            X_static_masked[primary_node, :-2] = merged_static
            if raw_has_indicators:
                X_static_raw_masked[primary_node, :-2] = merged_raw_static
            else:
                X_static_raw_masked[primary_node, :] = merged_raw_static
            X_static_masked[primary_node, -2] = 0.0
            X_static_masked[primary_node, -1] = 1.0

    if len(mask_indices) > 0:
        # 평가 대상 동은 사업체/종사자 관련 4개 컬럼만 마스킹한다.
        # zero-imputation이면 0, mean-imputation이면 train 평균으로 채운다.
        # 여기서 행 전체 feature를 0으로 만들면 원단위법/중력모델의 총량 예측이 무너진다.
        X_static_masked[np.ix_(mask_indices, masking_indices)] = scaled_impute_values
        X_static_raw_masked[np.ix_(mask_indices, masking_indices)] = raw_impute_values

    base_mask = mask | hide_mask
    raw_has_indicators = (X_static_raw_masked.shape[1] == X_static_masked.shape[1])
    
    if np.any(hide_mask):
        # hide_indices는 validation에서 치팅 방지를 위해 완전히 숨기는 동이다.
        # test split에서는 hide_indices=[]가 넘어와야 하므로 이 블록이 실행되면 안 된다.
        X_static_masked[hide_mask, :-2] = 0.0
        
        if raw_has_indicators:
            X_static_raw_masked[hide_mask, :-2] = 0.0
        else:
            X_static_raw_masked[hide_mask, :] = 0.0

    # indicator 컬럼 처리
    X_static_masked[base_mask, -2] = 1.0
    X_static_masked[base_mask, -1] = 0.0
    
    if raw_has_indicators:
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
    
    inactive_raw_fill = np.expm1(5.5)
    X_dist_curr_raw[inactive, :] = inactive_raw_fill
    X_dist_curr_raw[:, inactive] = inactive_raw_fill
    X_dist_curr_raw = np.where(np.isnan(X_dist_curr_raw), inactive_raw_fill, X_dist_curr_raw)


    out_X_static = torch.tensor(np.nan_to_num(X_static_masked, nan=0.0, posinf=0.0, neginf=0.0), dtype=torch.float32).clamp(-20.0, 20.0)
    out_X_static_raw = torch.tensor(np.nan_to_num(X_static_raw_masked, nan=0.0, posinf=0.0, neginf=0.0), dtype=torch.float32)
    out_X_dist = torch.tensor(np.nan_to_num(X_dist_curr, nan=5.5, posinf=5.5, neginf=5.5), dtype=torch.float32).clamp(0.0, 20.0)
    out_X_dist_raw = torch.tensor(np.nan_to_num(X_dist_curr_raw, nan=inactive_raw_fill, posinf=inactive_raw_fill, neginf=inactive_raw_fill), dtype=torch.float32)
    out_X_OD_masked = torch.tensor(np.nan_to_num(X_OD_masked, nan=0.0, posinf=0.0, neginf=0.0), dtype=torch.float32).clamp(0.0, 30.0)
    out_y_OD = torch.tensor(np.nan_to_num(y_OD, nan=0.0, posinf=0.0, neginf=0.0), dtype=torch.float32).clamp(0.0, 30.0)
    out_y_OD_raw = torch.tensor(np.nan_to_num(y_OD_raw, nan=0.0, posinf=0.0, neginf=0.0), dtype=torch.float32).clamp(0.0, 30.0)

    return {
        'X_static': out_X_static,
        'X_static_raw': out_X_static_raw,
        'X_dist': out_X_dist,
        'X_dist_raw': out_X_dist_raw,
        'X_OD_masked': out_X_OD_masked,
        'y_OD': out_y_OD,
        'y_OD_raw': out_y_OD_raw,
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
