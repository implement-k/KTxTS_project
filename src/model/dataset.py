import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TEST_CITIES_CODES, TRAIN_CONFIG, DONG_CODE_PATH,
    DIST_DATA_PATH, STATIC_DATA_PATH, OD_DATA_PATH, MASKING_COLUMNS
)

class ODDataset(Dataset):
    def __init__(self, data_dir=None, mode='train'):
        # data_dir is kept for backward compatibility but we use config paths
        self.mode = mode
        self.max_mask_size = TRAIN_CONFIG['max_mask_size']
        
        # 1. 행정동 표준 목록 로드 및 인덱스 맵 생성
        dong_df = pd.read_excel(DONG_CODE_PATH)
        valid_dongs = dong_df['dong_code'].astype(int).values
        self.num_nodes = len(valid_dongs)
        idx_map = {code: i for i, code in enumerate(valid_dongs)}
        
        # 2. OD 매트릭스 로드 및 피벗 (N, N, 5)
        print("Loading and pivoting OD data...")
        self.X_OD = np.zeros((self.num_nodes, self.num_nodes, 5), dtype=np.float32)
        od_df = pd.read_csv(OD_DATA_PATH)
        
        o_indices = od_df['O_dong_code'].map(idx_map).values
        d_indices = od_df['D_dong_code'].map(idx_map).values
        valid_mask = pd.notna(o_indices) & pd.notna(d_indices)
        
        o_idx_valid = o_indices[valid_mask].astype(int)
        d_idx_valid = d_indices[valid_mask].astype(int)
        
        purposes = ['귀가', '출근', '등교', '업무', '기타']
        for c, purpose in enumerate(purposes):
            if purpose in od_df.columns:
                self.X_OD[o_idx_valid, d_idx_valid, c] = od_df[purpose].values[valid_mask]
        
        # 3. 거리 매트릭스 로드 및 피벗 (N, N)
        print("Loading and pivoting Distance data...")
        self.X_dist = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        dist_df = pd.read_csv(DIST_DATA_PATH)
        
        o_dist = dist_df['O_dong_code'].map(idx_map).values
        d_dist = dist_df['D_dong_code'].map(idx_map).values
        dist_mask = pd.notna(o_dist) & pd.notna(d_dist)
        
        self.X_dist[o_dist[dist_mask].astype(int), d_dist[dist_mask].astype(int)] = dist_df['distance'].values[dist_mask]
        
        # 4. Static Feature 로드 및 정렬 매칭
        print("Loading Static Features...")
        static_df = pd.read_csv(STATIC_DATA_PATH)
        static_df['dong_code'] = static_df['dong_code'].astype(int)
        # valid_dongs 순서대로 완벽히 재정렬
        static_df = static_df.set_index('dong_code').reindex(valid_dongs).reset_index()
        static_df.fillna(0, inplace=True)
        
        drop_cols = ['dong_code', 'dong_name', 'sigungu_code', 'OD행정동코드', '행정동명']
        feature_cols = [c for c in static_df.columns if c not in drop_cols]
        
        # 마스킹할 컬럼의 정확한 인덱스 탐색
        self.masking_indices = [feature_cols.index(c) for c in MASKING_COLUMNS if c in feature_cols]
        raw_static = static_df[feature_cols].values
        
        # 피처 정규화 (StandardScaler)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        self.X_static = scaler.fit_transform(raw_static)
        
        # 마스킹 여부를 알려주는 Indicator 컬럼 추가 (0.0으로 초기화)
        indicator = np.zeros((self.X_static.shape[0], 1), dtype=np.float32)
        self.X_static = np.concatenate([self.X_static, indicator], axis=1)
            
        # 선택한 도시의 인덱스 찾기
        self.test_indices = self._find_dong_indices(idx_map)
        
        # Test 도시의 지정된 Feature 결측 처리 (0으로 마스킹)
        # target leakage 방지를 위해 예측 대상 도시의 실측 데이터는 모델에 제공하지 않습니다.
        for m_idx in self.masking_indices:
            self.X_static[self.test_indices, m_idx] = 0.0
        self.X_static[self.test_indices, -1] = 1.0 # is_masked = 1
        
        # train test 분리
        self.all_indices = np.arange(self.num_nodes)
        self.train_indices = np.setdiff1d(self.all_indices, self.test_indices)
        
        # 정규화 (거리 및 통행량 로그 변환)
        self.X_dist = np.log1p(self.X_dist)
        self.X_OD = np.log1p(self.X_OD)
        
        print("Dataset initialization complete.")
        
    def _find_dong_indices(self, idx_map):
        test_city_indices = []
        for _, codes in TEST_CITIES_CODES.items():
            for str_code in codes:
                code_int = int(str_code)
                if code_int in idx_map:
                    test_city_indices.append(idx_map[code_int])
        return np.array(test_city_indices)
        
    def __len__(self):
        # train 모드에서는 한 epoch당 1000번, test 모드에선 1개의 샘플만 반환
        return 1000 if self.mode == 'train' else 1

    def __getitem__(self, idx):
        if self.mode == 'train':
            # 마스크 사이즈 랜덤 선택
            k = np.random.randint(2, self.max_mask_size + 1)
    
            # train 동 중에서 한개 선택 후 그 동과의 거리 리스트 추출    
            dist_list = self.X_dist[np.random.choice(self.train_indices)]
            # 그중 test 동은 제외
            valid_distances = dist_list[self.train_indices]
            closest_k_indices = self.train_indices[np.argsort(valid_distances)[:k]]
            
            mask_indices = closest_k_indices
        else:
            mask_indices = self.test_indices
            
        # boolean mask (N,)
        mask = np.zeros(self.num_nodes, dtype=bool)
        mask[mask_indices] = True
        
        # Target OD (N, N, 5)
        y_OD = self.X_OD.copy()
        
        # Masked OD (해당 존의 출발/도착 통행량 모두 0으로 은닉)
        X_OD_masked = self.X_OD.copy()
        X_OD_masked[mask, :, :] = 0
        X_OD_masked[:, mask, :] = 0
        
        # Static Feature Dynamic Masking
        X_static_masked = self.X_static.copy()
        for m_idx in self.masking_indices:
            X_static_masked[mask, m_idx] = 0.0
        X_static_masked[mask, -1] = 1.0 # is_masked indicator
        
        return {
            'X_static': torch.tensor(X_static_masked, dtype=torch.float32),
            'X_dist': torch.tensor(self.X_dist, dtype=torch.float32),
            'X_OD_masked': torch.tensor(X_OD_masked, dtype=torch.float32),
            'y_OD': torch.tensor(y_OD, dtype=torch.float32),
            'mask': torch.tensor(mask, dtype=torch.bool)
        }