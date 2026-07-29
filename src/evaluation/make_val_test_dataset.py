import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'eval_common'))
import numpy as np
import torch
from config import VAL_CITIES_19_CODES, TEST_CITIES_19_CODES, VAL_CITIES_23_CODES, TEST_CITIES_23_CODES
import pandas as pd
from config import DONG_CODE_19_PATH, DONG_CODE_23_PATH
import importlib.util as _ilu
from fixed_eval_utils import make_base_data  # 신규 모듈

_mae_old_dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'mae-old', 'dataset.py')
_spec = _ilu.spec_from_file_location('mae_old_dataset', _mae_old_dataset_path)
if _spec is None:
    raise ImportError(f"Cannot find module 'mae_old_dataset' at {_mae_old_dataset_path}")
_mae_old_dataset = _ilu.module_from_spec(_spec)
if _spec.loader is None:
    raise ImportError(f"Cannot load module 'mae_old_dataset' from {_mae_old_dataset_path}")
_spec.loader.exec_module(_mae_old_dataset)
ODDataset = _mae_old_dataset.ODDataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
def generate_sample_meta_for_city(dataset, mask_indices, task, num_seeds=50):
    """메타 데이터만 생성"""
    metas = []
    mask_indices_set = set(mask_indices)
    N = dataset.num_nodes

    hide_mask = np.zeros(N, dtype=bool)
    if dataset.mode == 'val':
        if len(dataset.test_indices) > 0:
            hide_mask[dataset.test_indices] = True

    for seed in range(num_seeds):
        np.random.seed(seed + 1000 * task + hash(tuple(mask_indices)) % 10000)

        merge_events = []

        # 1. known merges (배경 병합) - task==0이면 아예 없음
        p_known_merges = 0.5
        if task != 0 and len(dataset.adjacency_candidates) > 0 and np.random.rand() < p_known_merges:
            n_known_merges = np.random.randint(1, 30 + 1)
            chosen_idxs = np.random.choice(len(dataset.adjacency_candidates),
                                            size=min(n_known_merges, len(dataset.adjacency_candidates)),
                                            replace=False)
            for i in chosen_idxs:
                a, b = dataset.adjacency_candidates[i]
                if a not in mask_indices_set and b not in mask_indices_set:
                    if not hide_mask[a] and not hide_mask[b]:  
                        merge_events.append((a, b, 'known_merge'))

        num_target_merges = np.random.randint(2, 4) if len(mask_indices) > 1 else 0

        mask_adj_candidates = [
            (a, b) for a in mask_indices for b in dataset.adj_list[a]
            if b in mask_indices_set and a < b
        ]
        known_adj_candidates = [
            (a, b) for a in mask_indices for b in dataset.adj_list[a]
            if b not in mask_indices_set and not hide_mask[b]
        ]

        if task == 0 or task == 1:
            pass
        elif task == 2:
            if len(known_adj_candidates) > 0:
                chosen = np.random.choice(len(known_adj_candidates), min(num_target_merges, len(known_adj_candidates)), replace=False)
                for i in chosen:
                    a, b = known_adj_candidates[i]
                    merge_events.append((a, b, 'mask_with_known'))
        elif task == 3:
            if len(mask_adj_candidates) > 0:
                chosen = np.random.choice(len(mask_adj_candidates), min(num_target_merges, len(mask_adj_candidates)), replace=False)
                for i in chosen:
                    a, b = mask_adj_candidates[i]
                    merge_events.append((a, b, 'mask_with_mask'))
        elif task == 4:
            all_cands = [('mask_with_known', c) for c in known_adj_candidates] + [('mask_with_mask', c) for c in mask_adj_candidates]
            if len(all_cands) > 0:
                chosen_idxs = np.random.choice(len(all_cands), min(num_target_merges, len(all_cands)), replace=False)
                for i in chosen_idxs:
                    etype, (a, b) = all_cands[i]
                    merge_events.append((a, b, etype))

        # 실제로 merge_cache에 존재하는 이벤트만 남겨서 저장 (재구성 시 불필요한 재검증 없이 바로 적용)
        used_b_nodes = set()
        applied_events = []
        for idx_a, idx_b, event_type in merge_events:
            if idx_a in used_b_nodes or idx_b in used_b_nodes:
                continue
            if event_type == 'mask_with_known':
                primary_node, secondary_node = (idx_a, idx_b) if idx_b in mask_indices_set else (idx_b, idx_a)
            else:
                primary_node, secondary_node = (idx_a, idx_b)
            cache_key = (primary_node, secondary_node)
            if cache_key not in dataset.merge_cache:
                cache_key = (secondary_node, primary_node)
                if cache_key not in dataset.merge_cache:
                    continue
            used_b_nodes.add(secondary_node)
            applied_events.append((idx_a, idx_b, event_type))

        metas.append({
            'mask_indices': list(mask_indices),
            'merge_events': applied_events,
        })

    return metas


def main():
    print("val/test dataset 로드")
    val_datasets = [ODDataset(year='2019', mode='val'), ODDataset(year='2023', mode='val')]
    test_datasets = [ODDataset(year='2019', mode='test'), ODDataset(year='2023', mode='test')]
    DONG_CODE_PATHS = [DONG_CODE_19_PATH, DONG_CODE_23_PATH]
    VAL_CITIES_YEAR_CODES = [VAL_CITIES_19_CODES, VAL_CITIES_23_CODES]
    TEST_CITIES_YEAR_CODES = [TEST_CITIES_19_CODES, TEST_CITIES_23_CODES]
    year_labels = ['2019', '2023']

    dong_dfs = [pd.read_excel(path) for path in DONG_CODE_PATHS]
    dong_codes_list = [dong_df['dong_code'].astype(int).values for dong_df in dong_dfs]
    dong2idx_maps = [{code: i for i, code in enumerate(codes)} for codes in dong_codes_list]

    out_dir = os.path.join(os.path.dirname(__file__), '../../dataset/fixed_eval')
    os.makedirs(out_dir, exist_ok=True)

    # === base_data는 연도당 1번만 저장 (val/test가 원본 데이터는 동일함) ===
    for i, year in enumerate(year_labels):
        base_data = make_base_data(val_datasets[i])
        base_path = os.path.join(out_dir, f'base_data_{year}.pt')
        torch.save(base_data, base_path)
        print(f"Saved base_data: {base_path}")

    ############### validation set (메타데이터만) ###############
    for i, val_cities_codes in enumerate(VAL_CITIES_YEAR_CODES):
        val_result = {}
        for city_name, dong_codes in val_cities_codes.items():
            print(f"[val] 연도: {year_labels[i]}, 생성: {city_name}...")
            indices = val_datasets[i]._find_dong_indices(dong2idx_maps[i], {city_name: dong_codes}).tolist()
            val_result[city_name] = {}
            for task in [0, 1, 2, 3, 4]:
                metas = generate_sample_meta_for_city(val_datasets[i], indices, task=task)
                val_result[city_name][task] = metas

        val_path = os.path.join(out_dir, f'fixed_val_meta_{year_labels[i]}.pt')
        torch.save(val_result, val_path)
        print(f"Saved {val_path}")

    ################## test set (메타데이터만) #################
    for i, test_cities_codes in enumerate(TEST_CITIES_YEAR_CODES):
        test_result = {}
        for city_name, dong_codes in test_cities_codes.items():
            print(f"[test] 연도: {year_labels[i]}, 생성: {city_name}...")
            indices = test_datasets[i]._find_dong_indices(dong2idx_maps[i], {city_name: dong_codes}).tolist()
            test_result[city_name] = {}
            for task in [0, 1, 2, 3, 4]:
                metas = generate_sample_meta_for_city(test_datasets[i], indices, task=task)
                test_result[city_name][task] = metas

        test_path = os.path.join(out_dir, f'fixed_test_meta_{year_labels[i]}.pt')
        torch.save(test_result, test_path)
        print(f"Saved {test_path}")


if __name__ == '__main__':
    main()