import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

from config import (
    TEST_CITIES_CODES, TRAIN_CONFIG, DONG_CODE_PATH,
    DIST_DATA_PATH, STATIC_DATA_PATH, OD_DATA_PATH, MASKING_COLUMNS, DATA_DIR
)

class ODDataset(Dataset):
    def __init__(self, mode='train', use_nan_masking=False, use_log_transform=True, use_od=True, predict_only_masked=False, use_residual=False, region_name='seoul'):
        self.mode = mode
        self.region_name = region_name
        self.max_mask_size = TRAIN_CONFIG['min_mask_size']
        
        self.use_od = use_od
        self.use_nan_masking = use_nan_masking
        self.predict_only_masked = predict_only_masked
        self.use_residual = use_residual

        if region_name == 'seoul':
            self._init_seoul(use_nan_masking, use_log_transform)
        else:
            self._init_regional(region_name, use_log_transform)

    def _init_regional(self, region, use_log_transform):
        # 1. OD 데이터 로드
        od_path = os.path.join(DATA_DIR, 'processed', f'od_{region}.csv')
        od_df = pd.read_csv(od_path)
        
        zones = sorted(list(set(od_df['출발'].unique()) | set(od_df['도착'].unique())))
        self.num_nodes = len(zones)
        zone2idx = {z: i for i, z in enumerate(zones)}
        
        self.X_OD = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        o_idx = od_df['출발'].map(zone2idx).values
        d_idx = od_df['도착'].map(zone2idx).values
        purposes = ['귀가', '출근', '등교', '업무', '기타']
        
        available_purposes = [c for c in purposes if c in od_df.columns]
        if len(available_purposes) == 0:
            if '합계' in od_df.columns:
                calculated_total = od_df['합계']
            else:
                calculated_total = od_df.iloc[:, 2:].sum(axis=1)
        else:
            calculated_total = od_df[available_purposes].sum(axis=1)
        
        self.X_OD[o_idx, d_idx] = calculated_total.values
        
        # Distance 매트릭스 로드
        dist_path = os.path.join(DATA_DIR, 'processed', f'{region}_dist.csv')
        dist_df = pd.read_csv(dist_path, index_col=0)
        # Convert index and columns to string for safe matching
        dist_df.index = dist_df.index.astype(str)
        dist_df.columns = dist_df.columns.astype(str)
        str_zones = [str(z) for z in zones]
        
        # Extract the submatrix efficiently using reindex
        self.X_dist = dist_df.reindex(index=str_zones, columns=str_zones, fill_value=0.0).values.astype(np.float32)
        # Static Features 로드 (타 지역은 데이터가 없으므로 0으로 패딩)
        seoul_static = pd.read_csv(STATIC_DATA_PATH)
        feature_cols = [c for c in seoul_static.columns if c not in ['dong_code', 'dong_name'] and not c.startswith('station_density') and c not in ['worker_density', 'business_density']]
        
        F = len(feature_cols) + 3 # density 3개 추가
        self.masking_indices = [0, 1] 
        self.X_static = np.zeros((self.num_nodes, F + 1), dtype=np.float32)
        
        self.test_indices = np.array([], dtype=int)
        self.all_indices = np.arange(self.num_nodes)
        self.train_indices = self.all_indices
        
        if use_log_transform:
            self.X_dist = np.log1p(self.X_dist)
            self.X_OD = np.log1p(self.X_OD)
            
        print(f"[{region}] Dataset 초기화 완료 (N={self.num_nodes})")

    def _init_seoul(self, use_nan_masking, use_log_transform):
        dong_df = pd.read_excel(DONG_CODE_PATH)
        dongs = dong_df['dong_code'].astype(int).values
        self.num_nodes = len(dongs)
        dong2idx_map = {code: i for i, code in enumerate(dongs)}
        
        self.X_OD = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        od_df = pd.read_csv(OD_DATA_PATH)
        
        o_indices = od_df['O_dong_code'].map(dong2idx_map).values
        d_indices = od_df['D_dong_code'].map(dong2idx_map).values
        valid_mask = pd.notna(o_indices) & pd.notna(d_indices)
        
        o_idx_valid = o_indices[valid_mask].astype(int)
        d_idx_valid = d_indices[valid_mask].astype(int)
        
        purposes = ['귀가', '출근', '등교', '업무', '기타']
        calculated_total = od_df[purposes].sum(axis=1)
        self.X_OD[o_idx_valid, d_idx_valid] = calculated_total.values[valid_mask]
        
        self.X_dist = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        dist_df = pd.read_csv(DIST_DATA_PATH)
        
        o_dist = dist_df['O_dong_code'].map(dong2idx_map).values
        d_dist = dist_df['D_dong_code'].map(dong2idx_map).values
        dist_mask = pd.notna(o_dist) & pd.notna(d_dist)
        
        self.X_dist[o_dist[dist_mask].astype(int), d_dist[dist_mask].astype(int)] = dist_df['distance'].values[dist_mask]
        
        static_df = pd.read_csv(STATIC_DATA_PATH)
        static_df['dong_code'] = static_df['dong_code'].astype(int)
        
        static_df = static_df.set_index('dong_code').reindex(dongs).reset_index()
        static_df.fillna(0, inplace=True)
        
        static_df['worker_density'] = static_df['worker_count'] / (static_df['행정동전체면적_m2'] + 1e-5)
        static_df['business_density'] = static_df['business_count'] / (static_df['행정동전체면적_m2'] + 1e-5)
        static_df['station_density_지하철'] = static_df['station_count_지하철'] / (static_df['행정동전체면적_m2'] + 1e-5)
        
        feature_cols = [c for c in static_df.columns if c not in ['dong_code', 'dong_name']]
        
        self.masking_indices = [feature_cols.index(c) for c in MASKING_COLUMNS if c in feature_cols]
        raw_static = static_df[feature_cols].values
        self.test_indices = self._find_dong_indices(dong2idx_map)
        self.all_indices = np.arange(self.num_nodes)
        self.train_indices = np.setdiff1d(self.all_indices, self.test_indices)
        
        scaler = StandardScaler()
        scaler.fit(raw_static[self.train_indices])
        self.X_static = scaler.transform(raw_static)
        
        if use_nan_masking:
            self.X_static[np.ix_(self.test_indices, self.masking_indices)] = np.nan
        else:
            indicator = np.zeros((self.X_static.shape[0], 1), dtype=np.float32)
            self.X_static = np.concatenate([self.X_static, indicator], axis=1)
            self.X_static = self.mask_static_features(self.X_static, self.test_indices, self.masking_indices)
        
        if use_log_transform:
            self.X_dist = np.log1p(self.X_dist)
            self.X_OD = np.log1p(self.X_OD)

        print("[seoul] Dataset 초기화 완료")
        
    def mask_static_features(self, X_static, mask_row_indices, mask_col_indices):
        X_masked = X_static.copy()
        X_masked[np.ix_(mask_row_indices, mask_col_indices)] = 0.0
        X_masked[mask_row_indices, -1] = 1.0
        return X_masked
        
    def _find_dong_indices(self, idx_map):
        indices = []
        for city, dongs in TEST_CITIES_CODES.items():
            for code in dongs:
                if int(code) in idx_map:
                    indices.append(idx_map[int(code)])
                else:
                    print(f"Warning: {code} not found in dong list")
        return np.array(indices)
        
    def __len__(self):
        return 1000 if self.mode == 'train' else 1

    def __getitem__(self, idx):
        if self.mode == 'train':
            k = np.random.randint(1, self.max_mask_size + 1)
            dist_list = self.X_dist[np.random.choice(self.train_indices)]
            valid_distances = dist_list[self.train_indices]
            closest_k_indices = self.train_indices[np.argsort(valid_distances)[:k]]
            mask_indices = closest_k_indices
        else:
            mask_indices = self.test_indices
            
        mask = np.zeros(self.num_nodes, dtype=bool)
        mask[mask_indices] = True
        
        y_OD = self.X_OD.copy()
        
        X_OD_masked = self.X_OD.copy()
        if len(self.test_indices) > 0:
            X_OD_masked[self.test_indices, :] = 0
            X_OD_masked[:, self.test_indices] = 0
            
        X_OD_masked[mask, :] = 0
        X_OD_masked[:, mask] = 0
        
        return {
            'X_static': torch.tensor(self.X_static, dtype=torch.float32),
            'X_dist': torch.tensor(self.X_dist, dtype=torch.float32),
            'X_OD_masked': torch.tensor(X_OD_masked, dtype=torch.float32),
            'y_OD': torch.tensor(y_OD, dtype=torch.float32),
            'mask': torch.tensor(mask, dtype=torch.bool)
        }

    def get_stage1_training_data(self, val_indices):
        train_mask = np.ones(self.num_nodes, dtype=bool)
        train_mask[self.test_indices] = False
        if val_indices is not None:
            train_mask[val_indices] = False
            
        x_od = self.X_OD.copy()
        x_od[:, ~train_mask] = 0 
        x_od[~train_mask, :] = 0 
        
        X_static_masked = self.X_static.copy()
        val_mask_np = np.zeros(self.num_nodes, dtype=bool)
        if val_indices is not None:
            val_mask_np[val_indices] = True
            X_static_masked = self.mask_static_features(X_static_masked, val_indices, self.masking_indices)
            x_od[val_mask_np, :] = 0
            x_od[:, val_mask_np] = 0
            
        x_s = torch.tensor(X_static_masked, dtype=torch.float32).unsqueeze(0)
        x_d = torch.tensor(self.X_dist, dtype=torch.float32).unsqueeze(0)
        y_od = torch.tensor(self.X_OD, dtype=torch.float32).unsqueeze(0)
        
        val_mask_1d_tensor = torch.tensor(val_mask_np, dtype=torch.bool).unsqueeze(0)
        val_mask_2d = val_mask_1d_tensor.unsqueeze(1) | val_mask_1d_tensor.unsqueeze(2)
        
        if self.use_log_transform:
            y_od_log = torch.log1p(y_od)
        else:
            y_od_log = y_od
            
        if val_indices is not None:
            return X_static_masked, x_s, x_d, y_od, y_od_log, val_mask_1d_tensor, val_mask_2d
        else:
            return X_static_masked, x_s, x_d, y_od, y_od_log, val_mask_2d

class MultiRegionDataset(Dataset):
    def __init__(self, regions, batch_size=32, mode='train', use_log_transform=True):
        self.regions = regions
        self.batch_size = batch_size
        self.mode = mode
        
        self.datasets = {}
        for r in regions:
            self.datasets[r] = ODDataset(mode=mode, use_log_transform=use_log_transform, region_name=r)
            
        # 서울 아닌 지역은 static feature 0으로 설정
        F = self.datasets['seoul'].X_static.shape[1]
        for r in regions:
            if r != 'seoul':
                self.datasets[r].X_static = np.zeros((self.datasets[r].num_nodes, F), dtype=np.float32)
                
        self.length = 1000 if mode == 'train' else 1

    def __len__(self):
        return self.length

    @property
    def max_mask_size(self):
        return getattr(self, '_max_mask_size', 1)

    @max_mask_size.setter
    def max_mask_size(self, value):
        self._max_mask_size = value
        for ds in self.datasets.values():
            ds.max_mask_size = value
        
    def __getitem__(self, idx):
        # 학습 시에는 다른 지역의 od를 학습하도록
        if self.mode == 'train':
            if np.random.rand() < 0.5 or len(self.regions) == 1:
                region = 'seoul'
            else:
                other_regions = [r for r in self.regions if r != 'seoul']
                region = np.random.choice(other_regions)
        # test 시에는 수도권만 예측
        else:
            region = 'seoul'
            
        ds = self.datasets[region]
        
        b_x_static = []
        b_x_dist = []
        b_x_od_masked = []
        b_y_od = []
        b_mask = []
        b_has_static = []
        
        B = self.batch_size if self.mode == 'train' else 1
        
        for _ in range(B):
            item = ds.__getitem__(idx)
            b_x_static.append(item['X_static'])
            b_x_dist.append(item['X_dist'])
            b_x_od_masked.append(item['X_OD_masked'])
            b_y_od.append(item['y_OD'])
            b_mask.append(item['mask'])
            b_has_static.append(torch.tensor(region == 'seoul', dtype=torch.bool))
            
        return {
            'X_static': torch.stack(b_x_static),
            'X_dist': torch.stack(b_x_dist),
            'X_OD_masked': torch.stack(b_x_od_masked),
            'y_OD': torch.stack(b_y_od),
            'mask': torch.stack(b_mask),
            'has_static': torch.stack(b_has_static)
        }