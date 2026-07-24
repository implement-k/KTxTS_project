import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from config import VAL_CITIES_CODES, TEST_CITIES_CODES
import pandas as pd
from config import DONG_CODE_PATH
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'mae-old'))
from dataset import ODDataset

def generate_samples_for_city(dataset, mask_indices, task, num_seeds=50):
    samples = []
    
    # Pre-extract base components so we don't pollute the main dataset
    mask_indices_set = set(mask_indices)
    N = dataset.num_nodes
    
    hide_mask = np.zeros(N, dtype=bool)
    # val: test_indices는 항상 숨김 처리
    if dataset.mode == 'val':
        if len(dataset.test_indices) > 0:
            hide_mask[dataset.test_indices] = True
           
    # val/test 개수가 적으므로 평균 +- 표준편차 형태로 출력하기 위해 반복
    for seed in range(num_seeds):
        np.random.seed(seed + 1000 * task + hash(tuple(mask_indices)) % 10000)
        
        mask = np.zeros(N, dtype=bool)
        if len(mask_indices) > 0: mask[mask_indices] = True
            
        y_OD_raw = dataset.X_OD_raw.copy()
        X_static_masked = dataset.X_static.copy()
        X_dist_curr = dataset.X_dist.copy()
        active_node_mask = np.ones(N, dtype=bool)

        merge_events = []
        
        # 1. known merges (mask_with_known) - 확률 0.5 (task 0일 때는 병합 안 함)
        p_known_merges = 0.5
        if task != 0 and len(dataset.adjacency_candidates) > 0 and np.random.rand() < p_known_merges:
            n_known_merges = np.random.randint(1, 30 + 1)
            chosen_idxs = np.random.choice(len(dataset.adjacency_candidates), 
                                            size=min(n_known_merges, len(dataset.adjacency_candidates)), 
                                            replace=False)
            for i in chosen_idxs:
                a, b = dataset.adjacency_candidates[i]
                if a not in mask_indices_set and b not in mask_indices_set:
                    merge_events.append((a, b, 'known_merge'))
                        
        num_target_merges = np.random.randint(2, 4) if len(mask_indices) > 1 else 0
        
        # 마스크 + 마스크 병합 후보
        mask_adj_candidates = [
            (a, b) for a in mask_indices for b in dataset.adj_list[a]
            if b in mask_indices_set and a < b
        ]
        
        # 마스크 + 알려진 병합 후보
        known_adj_candidates = [
            (a, b) for a in mask_indices for b in dataset.adj_list[a]
            if b not in mask_indices_set and not hide_mask[b]
        ]
        
        # task0: 모든 병합 없이 단순 마스킹만 (known_merge도 없음)
        if task == 0:
            pass
        # task1: 그냥 마스킹 된동 예측(새로운 동에 신도시가 생기는 경우 재현, 기존에 하던 task, known merge는 있음)
        elif task == 1:
            pass 
        # task2: 마스킹 된 동 + 알려진 동 병합 (기존의 동 안에 신도시가 생기는 경우 재현)
        elif task == 2:
            if len(known_adj_candidates) > 0:
                chosen = np.random.choice(len(known_adj_candidates), min(num_target_merges, len(known_adj_candidates)), replace=False)
                for i in chosen:
                    a, b = known_adj_candidates[i]
                    merge_events.append((a, b, 'mask_with_known'))
        # task3: 마스킹 된 동 + 마스킹 된 동 병합 (신도시 동의 scale에 따라 일반화가 되는지 확인)
        elif task == 3:
            if len(mask_adj_candidates) > 0:
                chosen = np.random.choice(len(mask_adj_candidates), min(num_target_merges, len(mask_adj_candidates)), replace=False)
                for i in chosen:
                    a, b = mask_adj_candidates[i]
                    merge_events.append((a, b, 'mask_with_mask'))
        # task4: task2 + task3 혼합 (일반화 + 기존 동 안에 신도시 생기는 경우 재현)
        elif task == 4:
            all_cands = [('mask_with_known', c) for c in known_adj_candidates] + [('mask_with_mask', c) for c in mask_adj_candidates]
            if len(all_cands) > 0:
                chosen_idxs = np.random.choice(len(all_cands), min(num_target_merges, len(all_cands)), replace=False)
                for i in chosen_idxs:
                    etype, (a, b) = all_cands[i]
                    merge_events.append((a, b, etype))
                    
        # merge 적용
        used_b_nodes = set()
        actual_merges = {'known_merge': 0, 'mask_with_mask': 0, 'mask_with_known': 0}
        
        for idx_a, idx_b, event_type in merge_events:
            if idx_a in used_b_nodes or idx_b in used_b_nodes: continue

            # primary_node: 병합 후 남는 노드, secondary_node: 병합 후 제거되는 노드, (known, known), (known, mask), (mask, mask)
            if event_type == 'mask_with_known':
                primary_node, secondary_node = (idx_a, idx_b) if idx_b in mask_indices_set else (idx_b, idx_a)
            else:
                primary_node, secondary_node = (idx_a, idx_b)
                
            cache_key = (primary_node, secondary_node)
            if cache_key not in dataset.merge_cache:
                cache_key = (secondary_node, primary_node)
                if cache_key not in dataset.merge_cache:
                    continue
            
            actual_merges[event_type] += 1
                    
            cache = dataset.merge_cache[cache_key]

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
            merged_static = dataset.scaler.transform(merged_raw_static.reshape(1, -1))[0]
            
            merged_dist_row = cache['merged_dist_row_at_a']
            X_dist_curr[primary_node, :] = np.log1p(merged_dist_row)
            X_dist_curr[:, primary_node] = np.log1p(merged_dist_row)

            if event_type == 'known_merge':
                mask[primary_node] = False
                X_static_masked[primary_node, :-2] = merged_static
                X_static_masked[primary_node, -2] = 0.0
                X_static_masked[primary_node, -1] = 0.0

            elif event_type == 'mask_with_mask':
                mask[primary_node] = True
                X_static_masked[primary_node, :-2] = merged_static
                X_static_masked[primary_node, dataset.masking_indices] = 0.0
                X_static_masked[primary_node, -2] = 1.0
                X_static_masked[primary_node, -1] = 1.0

            elif event_type == 'mask_with_known':
                mask[primary_node] = False
                X_static_masked[primary_node, :-2] = merged_static
                X_static_masked[primary_node, -2] = 0.0
                X_static_masked[primary_node, -1] = 1.0

        if len(mask_indices) > 0:
            X_static_masked[np.ix_(mask_indices, dataset.masking_indices)] = 0.0
            
        base_mask = mask | hide_mask
        if np.any(hide_mask):
            X_static_masked[hide_mask, :-2] = 0.0
            
        X_static_masked[base_mask, -2] = 1.0
        X_static_masked[base_mask, -1] = 0.0

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

        print(f"    Seed {seed}: {actual_merges}")

        samples.append({
            'X_static': torch.tensor(X_static_masked, dtype=torch.float16),
            'X_dist': torch.tensor(X_dist_curr, dtype=torch.float16),
            'X_OD_masked': torch.tensor(X_OD_masked, dtype=torch.float16),
            'y_OD': torch.tensor(y_OD, dtype=torch.float16),
            'mask': torch.tensor(mask, dtype=torch.bool),
            'active_node_mask': torch.tensor(active_node_mask, dtype=torch.bool),
            'loss_mask': torch.tensor(mask.copy(), dtype=torch.bool),
            'merge_stats': actual_merges
        })
    return samples

def main():
    print("val/test dataset 로드")
    val_dataset = ODDataset(mode='val')
    test_dataset = ODDataset(mode='test')
    
    # 행정동 코드 매핑
    dong_df = pd.read_excel(DONG_CODE_PATH)
    dong_codes = dong_df['dong_code'].astype(int).values
    dong2idx_map = {code: i for i, code in enumerate(dong_codes)}
    
    out_dir = os.path.join(os.path.dirname(__file__), '../../dataset/fixed_eval')
    os.makedirs(out_dir, exist_ok=True)
    
    ############### validation set ###############
    val_result = {}
    for city_name, dong_codes in VAL_CITIES_CODES.items():
        print(f"생성: {city_name}...")
        indices = val_dataset._find_dong_indices(dong2idx_map, {city_name: dong_codes}).tolist()
        val_result[city_name] = {}
        for task in [0, 1, 2, 3, 4]:
            samples = generate_samples_for_city(val_dataset, indices, task=task)
            val_result[city_name][task] = samples
            
    val_path = os.path.join(out_dir, 'fixed_val_dataset.pt')
    torch.save(val_result, val_path)
    print(f"Saved {val_path}")
    #############################################
    
    ################## test set #################
    test_result = {}
    for city_name, dong_codes in TEST_CITIES_CODES.items():
        print(f"생성: {city_name}...")
        indices = test_dataset._find_dong_indices(dong2idx_map, {city_name: dong_codes}).tolist()
        test_result[city_name] = {}
        for task in [0, 1, 2, 3, 4]:
            samples = generate_samples_for_city(test_dataset, indices, task=task)
            test_result[city_name][task] = samples
            
    test_path = os.path.join(out_dir, 'fixed_test_dataset.pt')
    torch.save(test_result, test_path)
    print(f"Saved {test_path}")
    #############################################

if __name__ == '__main__':
    main()
