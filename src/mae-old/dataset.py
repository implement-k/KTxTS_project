import os
import sys
import numpy as np
import pandas as pd
import torch
import pickle
import random
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
        
        # 마스킹할 컬럼의 인덱스 탐색
        self.masking_indices = [feature_cols.index(c) for c in MASKING_COLUMNS if c in feature_cols]
        raw_static = static_df[feature_cols].values
        
        # === 선택한 도시의 인덱스 찾기 및 train/val/test 분리 ===
        self.test_indices = self._find_dong_indices(dong2idx_map, TEST_CITIES_CODES)
        self.val_indices = self._find_dong_indices(dong2idx_map, VAL_CITIES_CODES)
        self.all_indices = np.arange(self.num_nodes)
        
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
        is_merge = False
        is_mask = False
        
        if self.mode == 'train':
            p = random.random()
            if p < 0.5: is_mask, is_merge = True, True
            elif p < 0.75: is_mask, is_merge = False, True
            else: is_mask, is_merge = True, False
                
            if is_mask:
                k = np.random.randint(1, self.max_mask_size + 1)
                dist_list = self.X_dist[np.random.choice(self.train_indices)]
                valid_distances = dist_list[self.train_indices]
                mask_indices = self.train_indices[np.argsort(valid_distances)[:k]].tolist()
        else:
            is_mask = True
            mask_indices = self.test_indices.tolist()
            
        # Base arrays
        mask = np.zeros(self.num_nodes, dtype=bool)
        if len(mask_indices) > 0:
            mask[mask_indices] = True
            
        y_OD = self.X_OD.copy()
        y_OD_raw = self.X_OD_raw.copy()
        X_OD_masked = self.X_OD.copy()
        X_static_masked = self.X_static.copy()
        X_dist_curr = self.X_dist.copy()
        active_node_mask = np.ones(self.num_nodes, dtype=bool)
        
        # OD Masking
        if is_mask:
            X_OD_masked[mask, :] = 0
            X_OD_masked[:, mask] = 0
            # 100% Static Masking (except indicator)
            X_static_masked[mask, :-2] = 0.0
            X_static_masked[mask, -2] = 1.0 # is_masked indicator
            X_static_masked[mask, -1] = 0.0 # is_merged indicator
            
        # Merge Augmentation
        if is_merge and len(self.adjacency_candidates) > 0:
            idx_a, idx_b = random.choice(self.adjacency_candidates)
            cache = self.merge_cache[(idx_a, idx_b)]
            
            # 1. OD Merge (Self-loop rule)
            orig_od = y_OD # Note: log1p scale, wait.
            raw_od_a = y_OD_raw[idx_a, :]
            raw_od_b = y_OD_raw[idx_b, :]

            new_self_loop = (
                y_OD_raw[idx_a, idx_a] + y_OD_raw[idx_b, idx_b] +
                y_OD_raw[idx_a, idx_b] + y_OD_raw[idx_b, idx_a]
            )
            
            raw_y_od_row_a = raw_od_a + raw_od_b
            raw_y_od_col_a = y_OD_raw[:, idx_a] + y_OD_raw[:, idx_b]

            # Convert back to log1p
            y_OD[idx_a, :] = np.log1p(raw_y_od_row_a)
            y_OD[:, idx_a] = np.log1p(raw_y_od_col_a)
            y_OD[idx_a, idx_a] = np.log1p(new_self_loop)
            
            # train 모드에서 80% 확률로 병합된 노드를 마스킹, 20% 확률로 병합 근사치만 제공
            mask_merged_node = random.random() < 0.8
            
            if mask_merged_node:
                mask[idx_a] = True
                X_OD_masked[idx_a, :] = 0.0
                X_OD_masked[:, idx_a] = 0.0
            else:
                mask[idx_a] = False
                X_OD_masked[idx_a, :] = y_OD[idx_a, :]
                X_OD_masked[:, idx_a] = y_OD[:, idx_a]
            
            # 2. Distance Merge
            merged_dist = cache['merged_dist_row_at_a']
            X_dist_curr[idx_a, :] = np.log1p(merged_dist)
            X_dist_curr[:, idx_a] = np.log1p(merged_dist)
            
            # 3. Static Merge
            merged_raw_static = cache['merged_raw_static_at_a']
            # Scale it
            merged_static_scaled = self.scaler.transform(merged_raw_static.reshape(1, -1))[0]
            
            if mask_merged_node:
                # (1, 1): 병합 + 완전 마스킹
                X_static_masked[idx_a, :-2] = 0.0
                X_static_masked[idx_a, -2] = 1.0 # is_masked
                X_static_masked[idx_a, -1] = 1.0 # is_merged
            else:
                # (0, 1): 병합 근사 (Context)
                X_static_masked[idx_a, :-2] = merged_static_scaled
                X_static_masked[idx_a, -2] = 0.0 # is_masked
                X_static_masked[idx_a, -1] = 1.0 # is_merged
                
            # 4. Virtual Deletion for idx_b
            active_node_mask[idx_b] = False
            mask[idx_b] = False # Do not predict for deactivated node
            X_static_masked[idx_b, :-2] = 0.0
            X_static_masked[idx_b, -2] = 1.0 # 명시적으로 마스킹(삭제)되었음을 표시
            X_static_masked[idx_b, -1] = 0.0
            X_OD_masked[idx_b, :] = 0.0
            X_OD_masked[:, idx_b] = 0.0
            
            # Set distance to 5.5 (이 값은 log1p가 적용된 boundaries 상한값 5.5입니다. raw km가 아님!)
            X_dist_curr[idx_b, :] = 5.5
            X_dist_curr[:, idx_b] = 5.5
        
        return {
            'X_static': torch.tensor(X_static_masked, dtype=torch.float32),
            'X_dist': torch.tensor(X_dist_curr, dtype=torch.float32),
            'X_OD_masked': torch.tensor(X_OD_masked, dtype=torch.float32),
            'y_OD': torch.tensor(y_OD, dtype=torch.float32),
            'mask': torch.tensor(mask, dtype=torch.bool),
            'active_node_mask': torch.tensor(active_node_mask, dtype=torch.bool)
        }
        