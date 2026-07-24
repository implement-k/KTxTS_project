import os
import sys
import numpy as np
import pandas as pd
import torch
import pickle
from collections import deque
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TEST_CITIES_CODES, VAL_CITIES_CODES, TRAIN_CONFIG, DONG_CODE_PATH,
    DIST_DATA_PATH, STATIC_DATA_PATH, OD_DATA_PATH, MASKING_COLUMNS
)

class ODDataset(Dataset):
    def __init__(self, mode='train'):
        self.mode = mode
        self.max_mask_size = TRAIN_CONFIG['min_mask_size']
        
        # === 행정동 코드 로드 ===
        dong_df = pd.read_excel(DONG_CODE_PATH)
        dongs = dong_df['dong_code'].astype(int).values
        self.num_nodes = len(dongs)   # 전체 동 개수
        dong2idx_map = {code: i for i, code in enumerate(dongs)}
        
        # === OD 매트릭스 로드 (N,N) ===
        self.X_OD = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        od_df = pd.read_csv(OD_DATA_PATH)
        
        # OD 데이터에서 유효한 행정동만 필터링
        o_indices = od_df['O_dong_code'].map(dong2idx_map).values
        d_indices = od_df['D_dong_code'].map(dong2idx_map).values
        valid_mask = pd.notna(o_indices) & pd.notna(d_indices)
        
        o_idx_valid = np.asarray(o_indices[valid_mask].astype(int))
        d_idx_valid = np.asarray(d_indices[valid_mask].astype(int))

        purposes = ['귀가', '출근', '등교', '업무', '기타']
        calculated_total = od_df[purposes].sum(axis=1)

        self.X_OD[o_idx_valid, d_idx_valid] = np.asarray(calculated_total.values[valid_mask])
        
        # === 거리 매트릭스 로드 (N, N) ===
        self.X_dist = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        dist_df = pd.read_csv(DIST_DATA_PATH)
        
        # dist에서 유효한 행정동만 필터링
        o_dist = np.asarray(dist_df['O_dong_code'].map(dong2idx_map).values.astype(int))
        d_dist = np.asarray(dist_df['D_dong_code'].map(dong2idx_map).values.astype(int))
        dist_mask = pd.notna(o_dist) & pd.notna(d_dist)
        
        # 거리 매트릭스에 값 채우기
        self.X_dist[o_dist[dist_mask], d_dist[dist_mask]] = np.asarray(dist_df['distance'].values[dist_mask])
        
        # === Static Feature 로드 ===
        static_df = pd.read_csv(STATIC_DATA_PATH)
        static_df['dong_code'] = static_df['dong_code'].astype(int)
        
        # 행정동 코드 기준으로 결측치 0으로 채우기
        static_df = static_df.set_index('dong_code').reindex(dongs).reset_index()
        static_df.fillna(0, inplace=True)
        
        # 밀도 추가
        static_df['worker_density'] = static_df['worker_count'] / (static_df['행정동전체면적_m2'] + 1e-5)
        static_df['business_density'] = static_df['business_count'] / (static_df['행정동전체면적_m2'] + 1e-5)
        static_df['station_density_지하철'] = static_df['station_count_지하철'] / (static_df['행정동전체면적_m2'] + 1e-5)
        
        # 기타지역비율_pct 추가 (비율 총합 100% 맞추기 위함)
        static_df['기타지역비율_pct'] = 100.0 - (static_df['상업업무지역비율_pct'] + static_df['공공시설지역비율_pct'] + static_df['주거지역비율_pct'])
        static_df['기타지역비율_pct'] = static_df['기타지역비율_pct'].clip(lower=0.0)
                
        feature_cols = [c for c in static_df.columns if c not in ['dong_code', 'dong_name']]
        self.feature_cols = feature_cols
        self.masking_indices = [feature_cols.index(c) for c in MASKING_COLUMNS if c in feature_cols]
        raw_static = static_df[feature_cols].values
        
        # === 선택한 도시의 인덱스 찾기 및 train/val/test 분리 ===
        self.test_indices = self._find_dong_indices(dong2idx_map, TEST_CITIES_CODES)
        self.val_indices = self._find_dong_indices(dong2idx_map, VAL_CITIES_CODES)
        self.all_indices = np.arange(self.num_nodes)
        
        self.val_city_indices = {
            city: np.array([dong2idx_map[int(c)] for c in codes if int(c) in dong2idx_map])
            for city, codes in VAL_CITIES_CODES.items()
        }
        
        # Test와 Val을 제외한 나머지를 Train으로 설정
        exclude_indices = np.union1d(self.test_indices, self.val_indices)
        self.train_indices = np.setdiff1d(self.all_indices, exclude_indices)
        
        # 피처 정규화
        self.scaler = StandardScaler()
        self.scaler.fit(raw_static[self.train_indices])
        self.X_static = self.scaler.transform(raw_static)
        
        # 마스킹 여부(is_masked)와 병합 여부(is_merged)를 알려주는 2D Indicator 컬럼 추가 (0.0으로 초기화)
        indicator = np.zeros((self.X_static.shape[0], 2), dtype=np.float32)
        self.X_static = np.concatenate([self.X_static, indicator], axis=1)
        # Test 도시의 지정된 Feature 결측 처리 (0으로 마스킹)
        self.X_static = self.mask_static_features(self.X_static, self.test_indices, self.masking_indices)
        
        # 정규화 (거리 및 통행량 로그 변환)
        self.X_dist_raw = self.X_dist.copy()
        self.X_OD_raw = self.X_OD.copy()
        
        # --- Stratified Masking Weights ---
        # 노드별 최대 트래픽 크기에 비례하는 가중치 계산 (대형 통행망 오버샘플링)
        # 자기 자신(Self-loop) 혹은 타 노드와의 통행 중 가장 큰 값을 기준으로 가중치 산정
        max_node_traffic = np.maximum(self.X_OD_raw.max(axis=1), self.X_OD_raw.max(axis=0))
        # 1000 미만 통행량은 가중치 1.0, 그 이상은 스케일에 비례해 증가 (예: 7만 = 70배 가중치)
        self.node_weights = np.clip(max_node_traffic / 1000.0, 1.0, None)
        
        self.X_dist = np.log1p(self.X_dist)
        self.X_OD = np.log1p(self.X_OD)
            
        # Merge Cache Load
        cache_path = os.path.join(os.path.dirname(__file__), 'merge_cache.pkl')
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                self.merge_cache = pickle.load(f)
            self.adjacency_candidates = list(self.merge_cache.keys())
        else:
            self.merge_cache = {}
            self.adjacency_candidates = []

        self.adj_list = [[] for _ in range(self.num_nodes)]
        for idx_a, idx_b in self.adjacency_candidates:
            self.adj_list[idx_a].append(idx_b)
            self.adj_list[idx_b].append(idx_a)
        for i in range(self.num_nodes):
            self.adj_list[i] = list(set(self.adj_list[i]))

        print("Dataset 초기화 완료")
        
    def mask_static_features(self, X_static, mask_row_indices, mask_col_indices):
        """
        Validation 도시의 종사자수, 사업체 수를 마스킹
        """
        X_masked = X_static.copy()
        X_masked[np.ix_(mask_row_indices, mask_col_indices)] = 0.0
        # -2: is_masked, -1: is_merged
        X_masked[mask_row_indices, -2] = 1.0
        X_masked[mask_row_indices, -1] = 0.0
        return X_masked
        
    def _find_dong_indices(self, idx_map, cities_codes):
        all_codes = (int(code) for codes in cities_codes.values() for code in codes)
        return np.array([idx_map[c] for c in all_codes if c in idx_map])
        
    def __len__(self):
        # train 모드에서는 한 epoch당 1000번, test 모드에선 1개의 샘플만 반환
        return 1000 if self.mode == 'train' else 1

    def __getitem__(self, idx):       
        mask_indices = []
        N = self.num_nodes

        if self.mode == 'train':
            available_train = set(self.train_indices)
            k = np.random.randint(1, self.max_mask_size + 1)
            
            if np.random.rand() < 0.5: num_chunks = 1
            else: num_chunks = np.random.randint(2, 6)
                
            num_chunks = min(num_chunks, len(available_train), k)
            if num_chunks > 0:
                avail_list = list(available_train)
                # 추출 시 가중치 반영 (Stratified Masking)
                weights = self.node_weights[avail_list]
                probs = weights / weights.sum()
                seeds = np.random.choice(avail_list, size=num_chunks, replace=False, p=probs)
            else:
                seeds = []
                
            mask_indices_set = set(seeds)
            queues = [deque([s]) for s in seeds]
            
            remaining_list = list(available_train - mask_indices_set)
            np.random.shuffle(remaining_list)
            
            while len(mask_indices_set) < k and available_train:
                added = False
                for q in queues:
                    if len(mask_indices_set) >= k:
                        break
                    if not q:
                        continue
                    
                    curr = q.popleft()
                    
                    neighbors = [n for n in self.adj_list[curr] if n in available_train and n not in mask_indices_set]
                    np.random.shuffle(neighbors)
                    
                    for n in neighbors:
                        mask_indices_set.add(n)
                        q.append(n)
                        added = True
                        if len(mask_indices_set) >= k:
                            break
                if not added:
                    # Clean up remaining_list lazily
                    while remaining_list and remaining_list[-1] in mask_indices_set:
                        remaining_list.pop()
                        
                    if not remaining_list:
                        break
                        
                    new_seed = remaining_list.pop()
                    mask_indices_set.add(new_seed)
                    queues.append(deque([new_seed]))
                    
            mask_indices = list(mask_indices_set)
        elif self.mode == 'val':
            mask_indices = self.val_indices.tolist()
        else:
            mask_indices = self.test_indices.tolist()
            
        mask_indices_set = set(mask_indices)
        
        # Base arrays
        mask = np.zeros(N, dtype=bool)
        if len(mask_indices) > 0: mask[mask_indices] = True
            
        hide_mask = np.zeros(N, dtype=bool)
        if self.mode == 'train':
            if len(self.val_indices) > 0:
                hide_mask[self.val_indices] = True
            if len(self.test_indices) > 0:
                hide_mask[self.test_indices] = True
        elif self.mode == 'val':
            if len(self.test_indices) > 0:
                hide_mask[self.test_indices] = True
            
        y_OD = self.X_OD.copy()
        y_OD_raw = self.X_OD_raw.copy()
        X_OD_masked = self.X_OD.copy()
        X_static_masked = self.X_static.copy()
        X_dist_curr = self.X_dist.copy()
        active_node_mask = np.ones(N, dtype=bool)
        
        # 2. 병합
        #   a. 알려진 동(마스킹 되지 않은 동)끼리 병합 - 정보 손실 없음(확률 0.3)
        #   b. 마스킹된 동끼리 병합 - 발생 안할 수 있음
        #   c. 마스킹된 동 + 알려진 동 병합 - 발생 안할 수 있음
        p_known_merges = 0.3
        max_known_merges = 30
        p_masked_merge = 0.5
        max_masked_merges = max(1, min(10, len(mask_indices)//3))

        merge_events = []  # [(idx_a, idx_b, event_type), ...]

        # a. 알려진 동끼리 병합 -- 확률적으로 발생, 발생 시 여러 쌍 가능
        if len(self.adjacency_candidates) > 0 and np.random.rand() < p_known_merges:
            n_known_merges = np.random.randint(1, max_known_merges + 1)
            chosen_idxs = np.random.choice(len(self.adjacency_candidates), 
                                            size=min(n_known_merges, len(self.adjacency_candidates)), 
                                            replace=False)
            for i in chosen_idxs:
                a, b = self.adjacency_candidates[i]
                if a not in mask_indices_set and b not in mask_indices_set:
                    if not hide_mask[a] and not hide_mask[b]:
                        merge_events.append((a, b, 'known_merge'))

        # b., c. 마스킹 클러스터 관련 병합
        if np.random.rand() < p_masked_merge and len(mask_indices) >= 1:
            n_masked_merges = np.random.randint(1, max_masked_merges + 1)
            for _ in range(n_masked_merges):
                sub_p = np.random.rand()

                # b. 마스킹된 두 동끼리 병합
                if sub_p < 0.5:
                    candidates = [
                        (a, b) for a in mask_indices for b in self.adj_list[a]
                        if b in mask_indices_set and a < b
                    ]
                    if candidates:
                        a, b = candidates[np.random.randint(len(candidates))]
                        merge_events.append((a, b, 'mask_with_mask'))
                # c. 마스킹된 동 + 알려진(비마스킹) 이웃 병합
                else: 
                    candidates = [
                        (a, b) for a in mask_indices for b in self.adj_list[a]
                        if b not in mask_indices_set and b not in self.val_indices and b not in self.test_indices
                    ]
                    if candidates:
                        a, b = candidates[np.random.randint(len(candidates))]
                        merge_events.append((a, b, 'mask_with_known'))

        # 3. static feature 마스킹 및 OD/거리 매트릭스 업데이트
        X_static_masked = self.X_static.copy()
        
        # 기본 마스킹: 특정(순수) 마스킹 노드들의 MASKING_COLUMNS만 0으로 처리 (면적 등은 유지)
        if len(mask_indices) > 0:
            X_static_masked[np.ix_(mask_indices, self.masking_indices)] = 0.0
            
        base_mask = mask | hide_mask
        if np.any(hide_mask):
            X_static_masked[hide_mask, :-2] = 0.0
            
        X_static_masked[base_mask, -2] = 1.0   # is_masked
        X_static_masked[base_mask, -1] = 0.0

        X_dist_curr = self.X_dist.copy()
        y_OD_raw = self.X_OD_raw.copy()

        used_b_nodes = set()  # 이미 병합되어 사라진 idx_b들 

        for idx_a, idx_b, event_type in merge_events:
            if idx_a in used_b_nodes or idx_b in used_b_nodes: continue  # 이미 다른 병합에 쓰인 노드는 건너뜀 

            if event_type == 'mask_with_known':
                primary_node, secondary_node = (idx_a, idx_b) if idx_b in mask_indices_set else (idx_b, idx_a)
            else:
                primary_node, secondary_node = (idx_a, idx_b)
                
            cache_key = (primary_node, secondary_node)
            if cache_key not in self.merge_cache:
                cache_key = (secondary_node, primary_node)
                if cache_key not in self.merge_cache:
                    continue
                    
            cache = self.merge_cache[cache_key]

            # self-loop 병합 (raw scale)
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
            merged_static = self.scaler.transform(merged_raw_static.reshape(1, -1))[0]
            
            merged_dist_row = cache['merged_dist_row_at_a']
            X_dist_curr[primary_node, :] = np.log1p(merged_dist_row)
            X_dist_curr[:, primary_node] = np.log1p(merged_dist_row)

            if event_type == 'known_merge':
                # 둘 다 알려짐 -> 진짜 합산된 실제 값, 정보 손실 없음
                mask[primary_node] = False
                X_static_masked[primary_node, :-2] = merged_static
                X_static_masked[primary_node, -2] = 0.0
                X_static_masked[primary_node, -1] = 0.0  # 근사치 아님, 실측 합산

            elif event_type == 'mask_with_mask':
                # 둘 다 모름 -> 병합 결과의 특정 컬럼만 마스킹
                mask[primary_node] = True
                X_static_masked[primary_node, :-2] = merged_static
                X_static_masked[primary_node, self.masking_indices] = 0.0
                X_static_masked[primary_node, -2] = 1.0
                X_static_masked[primary_node, -1] = 1.0

            elif event_type == 'mask_with_known':
                mask[primary_node] = False
                X_static_masked[primary_node, :-2] = merged_static
                X_static_masked[primary_node, -2] = 0.0
                X_static_masked[primary_node, -1] = 1.0  # 근사치임을 표시

        # 4. 최종 정답
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

        loss_mask = mask.copy()

        return {
            'X_static': torch.tensor(X_static_masked, dtype=torch.float32),
            'X_dist': torch.tensor(X_dist_curr, dtype=torch.float32),
            'X_OD_masked': torch.tensor(X_OD_masked, dtype=torch.float32),
            'y_OD': torch.tensor(y_OD, dtype=torch.float32),
            'mask': torch.tensor(mask, dtype=torch.bool),
            'active_node_mask': torch.tensor(active_node_mask, dtype=torch.bool),
            'loss_mask': torch.tensor(loss_mask, dtype=torch.bool),
        }