import os
import sys
import numpy as np
import pandas as pd
import torch, pickle
from collections import deque
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TEST_CITIES_23_CODES, TEST_CITIES_19_CODES, TRAIN_CONFIG, 
    DONG_CODE_23_PATH, DONG_CODE_19_PATH,
    DIST_DATA_23_PATH, DIST_DATA_19_PATH, 
    STATIC_DATA_23_PATH, STATIC_DATA_19_PATH, 
    OD_DATA_23_PATH, OD_DATA_19_PATH, 
    VAL_CITIES_23_CODES, VAL_CITIES_19_CODES,
    MASKING_COLUMNS,
)

def _coerce_numeric_raw_static(raw_static, expected_len):
    """merge_cache에 코드/지역명 같은 메타 컬럼이 섞여 있으면 숫자 feature만 추출한다."""
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
    raise ValueError(f"merged_raw_static에서 숫자 feature {expected_len}개를 만들 수 없음 (raw_len={values.size})")


# mae-year과 동일
# train 시에만 쓰이는 dataset 클래스
class ODDataset(Dataset):
    def __init__(self, year: str='2023', use_stratfied_masking=True, use_merge_train=True, 
                 use_od_log_transform=True, use_static_normalize=True, use_dist_log_transform=True):
        # self.mode = mode -> train에만 쓰이는 데이터셋
        self.year = year
        self.max_mask_size = TRAIN_CONFIG['min_mask_size']
        self.use_stratfied_masking = use_stratfied_masking
        self.use_merge_train = use_merge_train
        self.X_static_normalize = use_static_normalize
        self.use_od_log_transform = use_od_log_transform
        self.use_dist_log_transform = use_dist_log_transform
        
        # === 행정동 코드 로드 ===
        if self.year == '2019':
            DONG_CODE_PATH = DONG_CODE_19_PATH
            OD_DATA_PATH = OD_DATA_19_PATH
            DIST_DATA_PATH = DIST_DATA_19_PATH
            STATIC_DATA_PATH = STATIC_DATA_19_PATH
            TEST_CITIES_CODES = TEST_CITIES_19_CODES
            VAL_CITIES_CODES = VAL_CITIES_19_CODES
        else:
            DONG_CODE_PATH = DONG_CODE_23_PATH
            OD_DATA_PATH = OD_DATA_23_PATH
            DIST_DATA_PATH = DIST_DATA_23_PATH
            STATIC_DATA_PATH = STATIC_DATA_23_PATH
            TEST_CITIES_CODES = TEST_CITIES_23_CODES
            VAL_CITIES_CODES = VAL_CITIES_23_CODES

        # 행정동 코드 로드
        dong_df = pd.read_excel(DONG_CODE_PATH)
        dongs = dong_df['dong_code'].astype(int).values
        self.num_nodes = len(dongs)   # 전체 동 개수
        dong2idx_map = {code: i for i, code in enumerate(dongs)}
        
        # === OD 매트릭스 로드 (N,N) ===
        self.X_OD = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32) # raw OD
        od_df = pd.read_csv(OD_DATA_PATH)
        
        # OD 데이터에서 유효한 행정동만 필터링
        o_indices = od_df['O_dong_code'].map(dong2idx_map).values
        d_indices = od_df['D_dong_code'].map(dong2idx_map).values
        valid_mask = pd.notna(o_indices) & pd.notna(d_indices)
        
        o_idx_valid = np.asarray(o_indices[valid_mask].astype(int))
        d_idx_valid = np.asarray(d_indices[valid_mask].astype(int))
        
        # (N, N)으로 합산
        purposes = ['귀가', '출근', '등교', '업무', '기타']
        calculated_total = od_df[purposes].sum(axis=1)
        self.X_OD[o_idx_valid, d_idx_valid] = np.asarray(calculated_total.values[valid_mask]) # raw OD
        
        # === 거리 매트릭스 로드 (N, N) ===
        self.X_dist = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32) # raw dist
        dist_df = pd.read_csv(DIST_DATA_PATH)
        
        # dist에서 유효한 행정동만 필터링
        o_dist = np.asarray(dist_df['O_dong_code'].map(dong2idx_map).values)
        d_dist = np.asarray(dist_df['D_dong_code'].map(dong2idx_map).values)
        dist_mask = pd.notna(o_dist) & pd.notna(d_dist)
        
        # 거리 매트릭스에 값 채우기
        self.X_dist[o_dist[dist_mask].astype(int), d_dist[dist_mask].astype(int)] = np.asarray(dist_df['distance'].values[dist_mask]) # raw dist
        
        # === Static Feature 로드 ===
        static_df = pd.read_csv(STATIC_DATA_PATH)
        static_df['dong_code'] = static_df['dong_code'].astype(int)
        
        # 행정동 코드 기준으로 결측치 0으로 채우기
        static_df = static_df.set_index('dong_code').reindex(dongs).reset_index()
        static_df.fillna(0, inplace=True)
        
        feature_cols = [c for c in static_df.columns if c not in ['dong_code', 'dong_name']]
        feature_cols = sorted(feature_cols) # 정렬하여 일관성 유지
        print("I: static feature columns:", feature_cols)
        
        # 마스킹할 컬럼의 인덱스 탐색
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
        
        # test와 val을 제외한 나머지를 train으로 설정
        exclude_indices = np.union1d(self.test_indices, self.val_indices)
        self.train_indices = np.setdiff1d(self.all_indices, exclude_indices)
        
        # 피처 정규화
        self.scaler = StandardScaler()
        self.scaler.fit(raw_static[self.train_indices])
        
        self.X_static = self.scaler.transform(raw_static)    # norm static
        self.X_static_raw = raw_static.copy()           # raw static
        
        # 확장을 위한 코드 ##
        self.X_dist_raw = self.X_dist.copy()  # raw dist
        self.X_OD_raw = self.X_OD.copy()      # raw OD
        #####
        
        # --- Stratified Masking Weights ---
        # 노드별 최대 트래픽 크기에 비례하는 가중치 계산 (대형 통행망 오버샘플링)
        # 자기 자신(Self-loop) 혹은 타 노드와의 통행 중 가장 큰 값을 기준으로 가중치 산정
        max_node_traffic = np.maximum(self.X_OD_raw.max(axis=1), self.X_OD_raw.max(axis=0))
        # 1000 미만 통행량은 가중치 1.0, 그 이상은 스케일에 비례해 증가 (예: 7만 = 70배 가중치)
        self.node_weights = np.clip(max_node_traffic / 1000.0, 1.0, None)

        #### TODO 이거 쓸지 말지 결정
        # 마스킹 여부와 병합 여부를 알려주는 indicator ((0,1): 병합, (1,0): 마스킹, (1,1): 둘 다, (0,0): 둘 다 아님)
        indicator = np.zeros((self.X_static.shape[0], 2), dtype=np.float32)
        self.X_static = np.concatenate([self.X_static, indicator], axis=1)  # norm static
        self.X_static_raw = np.concatenate([self.X_static_raw, indicator], axis=1)  # raw static
        # Test 도시의 지정된 Feature 결측 처리 (0으로 마스킹)
        self.X_static = self.mask_static_features(self.X_static, self.test_indices, self.masking_indices)
        self.X_static_raw = self.mask_static_features(self.X_static_raw, self.test_indices, self.masking_indices)
        #########
        
        # 정규화 (거리 및 통행량 로그 변환)
        self.X_dist = np.log1p(self.X_dist)
        self.X_OD = np.log1p(self.X_OD)

        # merge cache 로드
        cache_path = os.path.join(os.path.dirname(__file__), f'merge_cache_{self.year}.pkl')
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                self.merge_cache = pickle.load(f)
            self.adjacency_candidates = list(self.merge_cache.keys())
        else:
            print("W: merge_cache 파일이 존재하지 않음. 빈 캐시로 초기화")
            self.merge_cache = {}
            self.adjacency_candidates = []
            
        self.adj_list = [[] for _ in range(self.num_nodes)]
        for idx_a, idx_b in self.adjacency_candidates:
            self.adj_list[idx_a].append(idx_b)
            self.adj_list[idx_b].append(idx_a)
        for i in range(self.num_nodes):
            self.adj_list[i] = list(set(self.adj_list[i]))
            
        # A_spatial 구성 (Geographical Adjacency)
        self.A_spatial = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        for idx_a, idx_b in self.adjacency_candidates:
            self.A_spatial[idx_a, idx_b] = 1.0
            self.A_spatial[idx_b, idx_a] = 1.0
        
        print("I: Dataset 초기화 완료")
        
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
        # train 모드에서는 한 epoch당 1000번
        return 1000

    def __getitem__(self, idx):
        mask_indices = []
        N = self.num_nodes

        available_train = set(self.train_indices)
        k = np.random.randint(1, self.max_mask_size + 1)
        
        if np.random.rand() < 0.5: num_chunks = 1
        else: num_chunks = np.random.randint(2, 6)
            
        num_chunks = min(num_chunks, len(available_train), k)
        if num_chunks > 0:
            avail_list = list(available_train)
            # 추출 시 가중치 반영 (Stratified Masking)
            if self.use_stratfied_masking:
                weights = self.node_weights[avail_list]
                probs = weights / weights.sum()
                seeds = np.random.choice(avail_list, size=num_chunks, replace=False, p=probs)
            else:
                seeds = np.random.choice(avail_list, size=num_chunks, replace=False)
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
        mask_indices_set = set(mask_indices)
        
        # Base arrays
        mask = np.zeros(N, dtype=bool)
        if len(mask_indices) > 0: mask[mask_indices] = True
            
        hide_mask = np.zeros(N, dtype=bool)
        if len(self.val_indices) > 0:
            hide_mask[self.val_indices] = True
        if len(self.test_indices) > 0:
            hide_mask[self.test_indices] = True
            
        y_OD = self.X_OD.copy()
        y_OD_raw = self.X_OD_raw.copy()
        X_OD_masked = self.X_OD.copy()
        X_static_masked = self.X_static.copy()
        X_static_raw_masked = self.X_static_raw.copy()
        X_dist_curr = self.X_dist.copy()
        X_dist_curr_raw = self.X_dist_raw.copy()
        A_spatial_curr = self.A_spatial.copy()
        active_node_mask = np.ones(N, dtype=bool)
        
        # 2. 병합
        #   a. 알려진 동(마스킹 되지 않은 동)끼리 병합 - 정보 손실 없음(확률 0.3)
        #   b. 마스킹된 동끼리 병합 - 발생 안할 수 있음
        #   c. 마스킹된 동 + 알려진 동 병합 - 발생 안할 수 있음
        merge_events = []  # [(idx_a, idx_b, event_type), ...]
        if self.use_merge_train:
            p_known_merges = 0.3
            max_known_merges = 30
            p_masked_merge = 0.5
            max_masked_merges = max(1, min(10, len(mask_indices)//3))

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
        
        # 기본 마스킹: 특정(순수) 마스킹 노드들의 MASKING_COLUMNS만 0으로 처리 (면적 등은 유지)
        if len(mask_indices) > 0:
            X_static_masked[np.ix_(mask_indices, self.masking_indices)] = 0.0
            X_static_raw_masked[np.ix_(mask_indices, self.masking_indices)] = 0.0
            
        base_mask = mask | hide_mask
        if np.any(hide_mask):
            X_static_masked[hide_mask, :-2] = 0.0
            X_static_raw_masked[hide_mask, :-2] = 0.0
            
        X_static_masked[base_mask, -2] = 1.0   # is_masked
        X_static_masked[base_mask, -1] = 0.0
        X_static_raw_masked[base_mask, -2] = 1.0
        X_static_raw_masked[base_mask, -1] = 0.0

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
            
            # A_spatial 갱신 (물리적 인접성 병합 및 대각선 0 초기화)
            A_spatial_curr[primary_node, :] = np.logical_or(A_spatial_curr[primary_node, :], A_spatial_curr[secondary_node, :]).astype(np.float32)
            A_spatial_curr[:, primary_node] = np.logical_or(A_spatial_curr[:, primary_node], A_spatial_curr[:, secondary_node]).astype(np.float32)
            A_spatial_curr[primary_node, primary_node] = 0.0

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

            F = self.scaler.mean_.shape[0]
            merged_raw_static = _coerce_numeric_raw_static(cache['merged_raw_static_at_a'], F)
            merged_static = self.scaler.transform(merged_raw_static.reshape(1, -1))[0]
            
            merged_dist_row = cache['merged_dist_row_at_a']
            X_dist_curr_raw[primary_node, :] = merged_dist_row       
            X_dist_curr_raw[:, primary_node] = merged_dist_row
            X_dist_curr[primary_node, :] = np.log1p(merged_dist_row)
            X_dist_curr[:, primary_node] = np.log1p(merged_dist_row)

            if event_type == 'known_merge':
                # 둘 다 알려짐 -> 진짜 합산된 실제 값, 정보 손실 없음
                mask[primary_node] = False
                X_static_masked[primary_node, :-2] = merged_static
                X_static_raw_masked[primary_node, :-2] = merged_raw_static
                X_static_masked[primary_node, -2] = 0.0
                X_static_masked[primary_node, -1] = 0.0  # 근사치 아님, 실측 합산
                X_static_raw_masked[primary_node, -2] = 0.0
                X_static_raw_masked[primary_node, -1] = 0.0

            elif event_type == 'mask_with_mask':
                # 둘 다 모름 -> 병합 결과의 특정 컬럼만 마스킹
                mask[primary_node] = True
                X_static_masked[primary_node, :-2] = merged_static
                X_static_raw_masked[primary_node, :-2] = merged_raw_static
                X_static_masked[primary_node, self.masking_indices] = 0.0
                X_static_raw_masked[primary_node, self.masking_indices] = 0.0
                X_static_masked[primary_node, -2] = 1.0
                X_static_masked[primary_node, -1] = 1.0
                X_static_raw_masked[primary_node, -2] = 1.0
                X_static_raw_masked[primary_node, -1] = 1.0

            elif event_type == 'mask_with_known':
                mask[primary_node] = False
                X_static_masked[primary_node, :-2] = merged_static
                X_static_raw_masked[primary_node, :-2] = merged_raw_static
                X_static_masked[primary_node, -2] = 0.0
                X_static_masked[primary_node, -1] = 1.0  # 근사치임을 표시
                X_static_raw_masked[primary_node, -2] = 0.0
                X_static_raw_masked[primary_node, -1] = 1.0

        # 4. 최종 정답
        y_OD = np.log1p(y_OD_raw)
        
        final_mask = mask | hide_mask

        X_OD_masked = y_OD.copy()
        X_OD_masked[final_mask, :] = 0.0
        X_OD_masked[:, final_mask] = 0.0
        for b in used_b_nodes:
            X_OD_masked[b, :] = 0.0
            X_OD_masked[:, b] = 0.0
            A_spatial_curr[b, :] = 0.0
            A_spatial_curr[:, b] = 0.0
            
        X_OD_masked_raw = y_OD_raw.copy()
        X_OD_masked_raw[final_mask, :] = 0.0
        X_OD_masked_raw[:, final_mask] = 0.0
        for b in used_b_nodes:
            X_OD_masked_raw[b, :] = 0.0
            X_OD_masked_raw[:, b] = 0.0

        inactive = ~active_node_mask
        X_dist_curr[inactive, :] = 5.5
        X_dist_curr[:, inactive] = 5.5
        X_dist_curr = np.where(np.isnan(X_dist_curr), 5.5, X_dist_curr)
        
        inactive_raw_fill = np.expm1(5.5)
        X_dist_curr_raw[inactive, :] = inactive_raw_fill
        
        out_X_static = X_static_masked if self.X_static_normalize else X_static_raw_masked
        out_X_dist = X_dist_curr if self.use_dist_log_transform else X_dist_curr_raw
        out_X_OD_masked = X_OD_masked if self.use_od_log_transform else X_OD_masked_raw
        out_y_OD = y_OD if self.use_od_log_transform else y_OD_raw
        
        out_X_static_t = torch.tensor(np.nan_to_num(out_X_static, nan=0.0, posinf=0.0, neginf=0.0), dtype=torch.float32)
        if self.X_static_normalize:
            out_X_static_t = out_X_static_t.clamp(-20.0, 20.0)
            
        out_X_dist_t = torch.tensor(np.nan_to_num(out_X_dist, nan=5.5, posinf=5.5, neginf=5.5), dtype=torch.float32).clamp(0.0, 20.0)
        out_X_OD_masked_t = torch.tensor(np.nan_to_num(out_X_OD_masked, nan=0.0, posinf=0.0, neginf=0.0), dtype=torch.float32).clamp(0.0, 30.0)
        out_y_OD_t = torch.tensor(np.nan_to_num(out_y_OD, nan=0.0, posinf=0.0, neginf=0.0), dtype=torch.float32).clamp(0.0, 30.0)
        
        out_A_spatial_t = torch.tensor(A_spatial_curr, dtype=torch.float32)
        
        return {
            'X_static': out_X_static_t,
            'X_dist': out_X_dist_t,
            'X_OD_masked': out_X_OD_masked_t,
            'A_spatial': out_A_spatial_t,
            'y_OD': out_y_OD_t,
            'mask': torch.tensor(mask, dtype=torch.bool),
            'active_node_mask': torch.tensor(active_node_mask, dtype=torch.bool),
            'loss_mask': torch.tensor(mask.copy(), dtype=torch.bool)
        }