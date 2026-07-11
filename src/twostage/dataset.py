import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TEST_CITIES_CODES, TRAIN_CONFIG, DONG_CODE_PATH,
    DIST_DATA_PATH, STATIC_DATA_PATH, OD_DATA_PATH, MASKING_COLUMNS
)

class ODDataset(Dataset):
    def __init__(self, mode='train'):
        self.mode = mode
        self.max_mask_size = TRAIN_CONFIG['min_mask_size']
        
        # 행정동 코드 로드
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
        
        o_idx_valid = o_indices[valid_mask].astype(int)
        d_idx_valid = d_indices[valid_mask].astype(int)
        
        
        # (N, N)으로 합산
        purposes = ['귀가', '출근', '등교', '업무', '기타']
        calculated_total = od_df[purposes].sum(axis=1)
        self.X_OD[o_idx_valid, d_idx_valid] = calculated_total.values[valid_mask]

        
        # === 거리 매트릭스 로드 (N, N) ===
        self.X_dist = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        dist_df = pd.read_csv(DIST_DATA_PATH)
        
        # dist에서 유효한 행정동만 필터링
        o_dist = dist_df['O_dong_code'].map(dong2idx_map).values
        d_dist = dist_df['D_dong_code'].map(dong2idx_map).values
        dist_mask = pd.notna(o_dist) & pd.notna(d_dist)
        
        # 거리 매트릭스에 값 채우기
        self.X_dist[o_dist[dist_mask].astype(int), d_dist[dist_mask].astype(int)] = dist_df['distance'].values[dist_mask]
        
        # === Static Feature 로드 ===
        static_df = pd.read_csv(STATIC_DATA_PATH)
        static_df['dong_code'] = static_df['dong_code'].astype(int)
        
        # 행정동 코드 기준으로 결측치 0으로 채우기
        static_df = static_df.set_index('dong_code').reindex(dongs).reset_index()
        static_df.fillna(0, inplace=True)
        
        feature_cols = [c for c in static_df.columns if c not in ['dong_code', 'dong_name']]
        
        # 마스킹할 컬럼의 인덱스 탐색
        self.masking_indices = [feature_cols.index(c) for c in MASKING_COLUMNS if c in feature_cols]
        raw_static = static_df[feature_cols].values
        # 선택한 도시의 인덱스 찾기 및 train/test 분리
        self.test_indices = self._find_dong_indices(dong2idx_map)
        self.all_indices = np.arange(self.num_nodes)
        self.train_indices = np.setdiff1d(self.all_indices, self.test_indices)
        
        # 피처 정규화
        scaler = StandardScaler()
        scaler.fit(raw_static[self.train_indices])
        self.X_static = scaler.transform(raw_static)
        
        # 마스킹 여부를 알려주는 Indicator 컬럼 추가 (0.0으로 초기화)
        indicator = np.zeros((self.X_static.shape[0], 1), dtype=np.float32)
        self.X_static = np.concatenate([self.X_static, indicator], axis=1)
        
        # Test 도시의 지정된 Feature 결측 처리 (0으로 마스킹)
        for m_idx in self.masking_indices:
            self.X_static[self.test_indices, m_idx] = 0.0
        self.X_static[self.test_indices, -1] = 1.0 # is_masked = 1
        
        ################### LGBM 위한 데이터셋 생성 ################### 
        # 1. Train 노드 마스킹 (동탄, 위례, 검단)
        train_mask = np.ones(self.num_nodes, dtype=bool)
        train_mask[self.test_indices] = False
        
        # 2. 데이터 파싱
        x_od = self.X_OD.copy()
        x_od[:, ~train_mask] = 0 # Test 도착 가리기
        x_od[~train_mask, :] = 0 # Test 출발 가리기
        
        # 2.1. 자기동 내부 통행량, 타 지역 간 통행량 계산
        y_self = np.diag(x_od) # 자기동 내부 통행량 (N,)
        y_inter = np.sum(x_od, axis=1) - y_self # 타 지역 간 통행량 (N,)
        
        # 3.  학습용 데이터셋 생성
        self.X_static_lgb = self.X_static[train_mask]
        self.y_self_train = y_self[train_mask]
        self.y_inter_train = y_inter[train_mask]
        ###########################################################
        
        print("Dataset 초기화 완료")
        
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
            k = np.random.randint(1, self.max_mask_size + 1)
    
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
        # mae1인 경우 (N,N)
        y_OD = self.X_OD.copy()
        
        # Masked OD 
        X_OD_masked = self.X_OD.copy()
        
        # mask
        X_OD_masked[mask, :] = 0
        X_OD_masked[:, mask] = 0

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