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
    def __init__(self, mode='train', use_nan_masking=False, use_log_transform=True, use_od=True, predict_only_masked=False, use_residual=False):
        self.mode = mode
        self.max_mask_size = TRAIN_CONFIG['min_mask_size']
        
        # twostage용 코드
        self.use_od = use_od
        self.use_nan_masking = use_nan_masking
        self.predict_only_masked = predict_only_masked
        self.use_residual = use_residual

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
        
        # 밀도 파생변수 추가
        static_df['worker_density'] = static_df['worker_count'] / (static_df['행정동전체면적_m2'] + 1e-5)
        static_df['business_density'] = static_df['business_count'] / (static_df['행정동전체면적_m2'] + 1e-5)
        static_df['station_density_지하철'] = static_df['station_count_지하철'] / (static_df['행정동전체면적_m2'] + 1e-5)
        
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
        
        if use_nan_masking: # twostage v3
            self.X_static[np.ix_(self.test_indices, self.masking_indices)] = np.nan
        else:
            # 마스킹 여부를 알려주는 Indicator 컬럼 추가 (0.0으로 초기화)
            indicator = np.zeros((self.X_static.shape[0], 1), dtype=np.float32)
            self.X_static = np.concatenate([self.X_static, indicator], axis=1)
            # Test 도시의 지정된 Feature 결측 처리 (0으로 마스킹)
            self.X_static = self.mask_static_features(self.X_static, self.test_indices, self.masking_indices)
        
        # 정규화 (거리 및 통행량 로그 변환)
        if use_log_transform:
            self.X_dist = np.log1p(self.X_dist)
            self.X_OD = np.log1p(self.X_OD)

        print("Dataset 초기화 완료")
        
    def mask_static_features(self, X_static, mask_row_indices, mask_col_indices):
        """
        Validation 도시의 종사자수, 사업체 수를 마스킹
        """
        X_masked = X_static.copy()
        X_masked[np.ix_(mask_row_indices, mask_col_indices)] = 0.0
        # 마지막 컬럼을 1로 세팅 (마스킹되었다는 플래그 역할)
        X_masked[mask_row_indices, -1] = 1.0
        return X_masked
        
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
        
        # (N,N)
        y_OD = self.X_OD.copy()
        
        # Masked OD 
        X_OD_masked = self.X_OD.copy()
        X_OD_masked[mask, :] = 0
        X_OD_masked[:, mask] = 0
        
        # Static Feature Dynamic Masking (실제 마스킹은 모델 내부에서 mask_token으로 대체됨)
        # Loss 계산을 위해 Ground Truth(원본) 그대로 반환
        
        return {
            'X_static': torch.tensor(self.X_static, dtype=torch.float32),
            'X_dist': torch.tensor(self.X_dist, dtype=torch.float32),
            'X_OD_masked': torch.tensor(X_OD_masked, dtype=torch.float32),
            'y_OD': torch.tensor(y_OD, dtype=torch.float32),
            'mask': torch.tensor(mask, dtype=torch.bool)
        }
        
    # twostage용 코드
    def get_stage1_training_data(self, val_indices):
        # Train 노드 마스킹 (동탄, 위례, 검단 등 + 현재 val_indices)
        train_mask = np.ones(self.num_nodes, dtype=bool)
        train_mask[self.test_indices] = False
        if val_indices is not None:
            train_mask[val_indices] = False
            
        x_od = self.X_OD.copy()
        x_od[:, ~train_mask] = 0 # Test 도착 가리기
        x_od[~train_mask, :] = 0 # Test 출발 가리기
        
        if not self.use_od: # v2, v3: 자기동 내부 통행량, 타 지역 간 통행량 계산 (use_od=False)
            y1 = np.diag(x_od) # y_self
            y2 = np.sum(x_od, axis=1) - y1 # y_inter
        else:   # v4, v5: 총 발생량, 총 도착량 계산 (use_od=True)
            y1 = np.sum(x_od, axis=1) # y_o
            y2 = np.sum(x_od, axis=0) # y_d
            
        X_static_lgb = self.X_static[train_mask]
        y1_train = y1[train_mask]
        y2_train = y2[train_mask]
        
        return X_static_lgb, y1_train, y2_train, train_mask

    def get_validation_data(self, val_indices):
        X_static_masked = self.X_static.copy()
        if self.use_nan_masking:
            X_static_masked[np.ix_(self.test_indices, self.masking_indices)] = np.nan
        else:
            self.mask_static_features(X_static_masked, val_indices, self.masking_indices)
            
        x_s = torch.tensor(X_static_masked, dtype=torch.float32).unsqueeze(0)
        x_d = torch.tensor(self.X_dist, dtype=torch.float32).unsqueeze(0)
        
        val_mask_1d = np.zeros(self.num_nodes, dtype=bool)
        val_mask_1d[val_indices] = True
        val_mask_1d_tensor = torch.tensor(val_mask_1d, dtype=torch.bool).unsqueeze(0)
        val_mask_2d = val_mask_1d_tensor.unsqueeze(1) | val_mask_1d_tensor.unsqueeze(2)
        
        y_od = torch.tensor(self.X_OD, dtype=torch.float32).unsqueeze(0)
        y_od_log = torch.log1p(y_od)
        
        if self.use_residual or self.predict_only_masked: 
            return X_static_masked, x_s, x_d, y_od, y_od_log, val_mask_1d_tensor, val_mask_2d
        else:
            return X_static_masked, x_s, x_d, y_od, y_od_log, val_mask_2d