import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

from config import (
    VAL_CITIES_CODES, TEST_CITIES_CODES, TRAIN_CONFIG, DONG_CODE_PATH,
    DIST_DATA_PATH, STATIC_DATA_PATH, OD_DATA_PATH, MASKING_COLUMNS, DATA_DIR
)

# static feature 숫자 종류에 따라 학습법 분류를 위해 col 정의
CONT_COLS = [
    '행정동전체면적_m2', 'pop_0_19', 'pop_20_59', 'pop_60_plus',
    'worker_count', 'business_count',
    'worker_density', 'business_density', 'station_density_지하철'
]
PROP_MULTI_COLS = ['상업업무지역비율_pct', '공공시설지역비율_pct', '주거지역비율_pct', '기타지역비율_pct']
PROP_SINGLE_COLS = ['아파트비율_퍼센트']
ZERO_COLS = [
    'station_count_준고속철도', 'station_count_고속철도',
    'station_count_지하철', 'station_count_일반철도'
]

class ODDataset(Dataset):
    def __init__(self, mode='train', use_od=True, predict_only_masked=False, use_residual=False, region_name='seoul'):
        self.mode = mode
        self.region_name = region_name
        self.max_mask_size = TRAIN_CONFIG['min_mask_size']  # 커리큘럼: train.py가 epoch마다 점증
        
        self.use_od = use_od
        self.predict_only_masked = predict_only_masked
        self.use_residual = use_residual

        if region_name == 'seoul':
            self._init_seoul(region_name)
        else:
            self._init_regional(region_name)

    # static feature 없는 지역(수도권 제외)
    def _init_regional(self, region: str) -> None:
        # od 데이터 로드, distance 매트릭스 로드
        od_df = pd.read_csv(os.path.join(DATA_DIR, 'processed', f'od_{region}.csv'))
        dist_df = pd.read_csv(os.path.join(DATA_DIR, 'processed', f'{region}_dist.csv'), index_col=0)
        
        ###### od 데이터 로드 ######
        zones = sorted(list(set(od_df['출발'].unique()) | set(od_df['도착'].unique())))
        self.num_dongs = len(zones)
        zone2idx = {z: i for i, z in enumerate(zones)}
        
        self.X_OD = np.zeros((self.num_dongs, self.num_dongs), dtype=np.float32)
        o_idx = od_df['출발'].map(zone2idx).values
        d_idx = od_df['도착'].map(zone2idx).values
        purposes = ['귀가', '출근', '등교', '업무', '기타']
        
        available_purposes = [c for c in purposes if c in od_df.columns]
        if len(available_purposes) == 0:
            if '합계' in od_df.columns:
                purpose_total = od_df['합계']
            else:
                purpose_total = od_df.iloc[:, 2:].sum(axis=1)
        else:
            purpose_total = od_df[available_purposes].sum(axis=1)
        
        self.X_OD[o_idx, d_idx] = purpose_total.values
        ###############
        
        ####### 거리행렬 로드 #######
        dist_df.index = dist_df.index.astype(str)
        dist_df.columns = dist_df.columns.astype(str)
        str_zones = [str(z) for z in zones]
        
        self.X_dist = dist_df.reindex(index=str_zones, columns=str_zones, fill_value=0.0).values.astype(np.float32)
        
        missing_zones = set(str_zones) - set(dist_df.index)
        print(f"D: [dataset] 거리행렬에 없는 zone 개수: {len(missing_zones)}")
        if missing_zones:
            print(f"D: [dataset] 거리행렬에 없는 zone: {missing_zones}")
        
        # Static Features 로드 (타 지역은 데이터가 없으므로 0으로 패딩)
        self.X_cont = np.zeros((self.num_dongs, len(CONT_COLS)), dtype=np.float32)
        self.X_prop_multi = np.zeros((self.num_dongs, len(PROP_MULTI_COLS)), dtype=np.float32)
        self.X_prop_single = np.zeros((self.num_dongs, len(PROP_SINGLE_COLS)), dtype=np.float32)
        self.X_zero = np.zeros((self.num_dongs, len(ZERO_COLS)), dtype=np.float32)
        
        self.val_indices = np.array([], dtype=int)
        self.test_indices = np.array([], dtype=int)
        self.all_indices = np.arange(self.num_dongs)
        self.train_indices = self.all_indices
        
        self.X_dist = np.log1p(self.X_dist)
        self.X_OD = np.log1p(self.X_OD)
        ###################
        print(f"D: [dataset] {region} 초기화 완료 (N={self.num_dongs})")

    # static feature 있는 지역(수도권)
    def _init_seoul(self, region: str) -> None:
        dong_df = pd.read_excel(DONG_CODE_PATH)
        dongs = dong_df['dong_code'].astype(int).values
        self.num_dongs = len(dongs)
        self.idx_map = {code: i for i, code in enumerate(dongs)}
        
        self.X_OD = np.zeros((self.num_dongs, self.num_dongs), dtype=np.float32)
        od_df = pd.read_csv(OD_DATA_PATH)
        
        o_indices = od_df['O_dong_code'].map(self.idx_map).values
        d_indices = od_df['D_dong_code'].map(self.idx_map).values
        valid_mask = pd.notna(o_indices) & pd.notna(d_indices)
        
        o_idx_valid = o_indices[valid_mask].astype(int)
        d_idx_valid = d_indices[valid_mask].astype(int)
        
        purposes = ['귀가', '출근', '등교', '업무', '기타']
        purpose_total = od_df[purposes].sum(axis=1)
        self.X_OD[o_idx_valid, d_idx_valid] = purpose_total.values[valid_mask]
        
        self.X_dist = np.zeros((self.num_dongs, self.num_dongs), dtype=np.float32)
        dist_df = pd.read_csv(DIST_DATA_PATH)
        
        o_dist = dist_df['O_dong_code'].map(self.idx_map).values
        d_dist = dist_df['D_dong_code'].map(self.idx_map).values
        dist_mask = pd.notna(o_dist) & pd.notna(d_dist)
        
        self.X_dist[o_dist[dist_mask].astype(int), d_dist[dist_mask].astype(int)] = dist_df['distance'].values[dist_mask]
        
        static_df = pd.read_csv(STATIC_DATA_PATH)
        static_df['dong_code'] = static_df['dong_code'].astype(int)
        
        static_df = static_df.set_index('dong_code').reindex(dongs).reset_index()
        static_df.fillna(0, inplace=True)
        
        static_df['worker_density'] = static_df['worker_count'] / (static_df['행정동전체면적_m2'] + 1e-5)
        static_df['business_density'] = static_df['business_count'] / (static_df['행정동전체면적_m2'] + 1e-5)
        static_df['station_density_지하철'] = static_df['station_count_지하철'] / (static_df['행정동전체면적_m2'] + 1e-5)
        
        # 파생 변수 및 비율 변수 조정
        static_df['기타지역비율_pct'] = 100.0 - (static_df['상업업무지역비율_pct'] + static_df['공공시설지역비율_pct'] + static_df['주거지역비율_pct'])
        static_df['기타지역비율_pct'] = static_df['기타지역비율_pct'].clip(lower=0.0)
        
        # 원본 인구수 (Physics Prior용)
        pop_total = static_df['pop_0_19'] + static_df['pop_20_59'] + static_df['pop_60_plus']
        self.pop_raw = np.log1p(pop_total.values.astype(np.float32))
        
        # Determine continuous feature indices to mask dynamically
        masking_targets = MASKING_COLUMNS + [c.replace('count', 'density') for c in MASKING_COLUMNS if 'count' in c]
        self.mask_cont_indices = [i for i, c in enumerate(CONT_COLS) if c in masking_targets]
        
        for c in PROP_MULTI_COLS + PROP_SINGLE_COLS:
            static_df[c] = static_df[c] / 100.0
            
        prop_sum = static_df[PROP_MULTI_COLS].sum(axis=1) + 1e-8
        for c in PROP_MULTI_COLS:
            static_df[c] = static_df[c] / prop_sum
            
        orig_val_indices = self._find_dong_indices(self.idx_map, VAL_CITIES_CODES)
        orig_test_indices = self._find_dong_indices(self.idx_map, TEST_CITIES_CODES)
        
        # Decide which to excise based on mode
        excise_indices = []
        if self.mode == 'train':
            excise_indices = np.union1d(orig_val_indices, orig_test_indices)
        elif self.mode == 'val':
            excise_indices = orig_test_indices
        elif self.mode == 'test':
            excise_indices = []
            
        # Excise from the graph
        if len(excise_indices) > 0:
            active_indices = np.setdiff1d(np.arange(self.num_dongs), excise_indices)
            
            # Map original indices to new active indices
            orig_to_new = {orig: new for new, orig in enumerate(active_indices)}
            
            self.val_indices = np.array([orig_to_new[idx] for idx in orig_val_indices if idx in orig_to_new], dtype=int)
            self.test_indices = np.array([orig_to_new[idx] for idx in orig_test_indices if idx in orig_to_new], dtype=int)
            
            # Reduce data
            static_df = static_df.iloc[active_indices].reset_index(drop=True)
            self.num_dongs = len(active_indices)
            self.X_OD = self.X_OD[np.ix_(active_indices, active_indices)]
            self.X_dist = self.X_dist[np.ix_(active_indices, active_indices)]
        else:
            self.val_indices = orig_val_indices
            self.test_indices = orig_test_indices
            
        self.train_indices = np.setdiff1d(np.arange(self.num_dongs), np.union1d(self.val_indices, self.test_indices))
        
        raw_cont = static_df[CONT_COLS].values
        self.scaler = StandardScaler()
        self.scaler.fit(raw_cont[self.train_indices])
        
        self.X_cont = self.scaler.transform(raw_cont).astype(np.float32)
        self.X_prop_multi = static_df[PROP_MULTI_COLS].values.astype(np.float32)
        self.X_prop_single = static_df[PROP_SINGLE_COLS].values.astype(np.float32)
        self.X_zero = static_df[ZERO_COLS].values.astype(np.float32)
        
        self.X_dist = np.log1p(self.X_dist)
        self.X_OD = np.log1p(self.X_OD)

        print(f"D: [dataset] {region} 초기화 완료")
        
    def _find_dong_indices(self, idx_map: dict, cities_dict: dict) -> np.ndarray:
        indices = []
        for _, dongs in cities_dict.items():
            for code in dongs:
                if int(code) in idx_map:
                    indices.append(idx_map[int(code)])
                else:
                    pass
        return np.array(indices)
        
    def __len__(self):
        return 200 if self.mode == 'train' else 1  # 32,000 → 6,400 samples/epoch (A100 최적화)

    def __getitem__(self, idx):
        '''
            return:
                - static features: (B, N, F) -
                X_cont: continuous static features
                X_prop_multi: multi-class proportion static features
                X_prop_single: single-class proportion static features
                X_zero: zero-valued static features
                
                - dynamic features: (B, N, N) -
                X_dist: distance matrix (log-scaled)
                X_OD_masked: OD matrix with masked nodes
                y_OD: ground truth OD matrix
                
                mask: (B, N) - boolean mask where True means masked (predict this)
        '''
        
        if self.mode == 'train':
            k = np.random.randint(1, self.max_mask_size + 1)
            dist_list = self.X_dist[np.random.choice(self.train_indices)]
            valid_distances = dist_list[self.train_indices]
            closest_k_indices = self.train_indices[np.argsort(valid_distances)[:k]]
            
            mask_indices = closest_k_indices
        elif self.mode == 'val':
            mask_indices = self.val_indices
        else:
            mask_indices = self.test_indices
            
        mask = np.zeros(self.num_dongs, dtype=bool)
        mask[mask_indices] = True
        
        y_OD = self.X_OD.copy()
        
        X_OD_masked = self.X_OD.copy()
        X_OD_masked[mask, :] = 0
        X_OD_masked[:, mask] = 0
        
        x_cont = torch.tensor(self.X_cont, dtype=torch.float32)
        x_prop_multi = torch.tensor(self.X_prop_multi, dtype=torch.float32)
        x_prop_single = torch.tensor(self.X_prop_single, dtype=torch.float32)
        x_zero = torch.tensor(self.X_zero, dtype=torch.float32)
        x_od_tensor = torch.tensor(X_OD_masked, dtype=torch.float32)
        y_od_tensor = torch.tensor(y_OD, dtype=torch.float32)
        x_dist_tensor = torch.tensor(self.X_dist, dtype=torch.float32)
        pop_raw_tensor = torch.tensor(self.pop_raw, dtype=torch.float32)
        
        return {
            'X_cont': x_cont,
            'X_prop_multi': x_prop_multi,
            'X_prop_single': x_prop_single,
            'X_zero': x_zero,
            'X_OD_masked': x_od_tensor,
            'X_dist': x_dist_tensor,
            'y_OD': y_od_tensor,
            'mask': torch.tensor(mask, dtype=torch.bool),
            'loss_mask': torch.tensor(mask, dtype=torch.bool),
            'pop_raw': pop_raw_tensor
        }

    def get_stage1_training_data(self, val_indices):
        train_mask = np.ones(self.num_dongs, dtype=bool)
        train_mask[self.test_indices] = False
        if val_indices is not None:
            train_mask[val_indices] = False
            
        x_od = self.X_OD.copy()
        x_od[:, ~train_mask] = 0 
        x_od[~train_mask, :] = 0 
        
        X_static_masked = self.X_static.copy()
        val_mask_np = np.zeros(self.num_dongs, dtype=bool)
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
        
        y_od_log = torch.log1p(y_od)
            
        if val_indices is not None:
            return X_static_masked, x_s, x_d, y_od, y_od_log, val_mask_1d_tensor, val_mask_2d
        else:
            return X_static_masked, x_s, x_d, y_od, y_od_log, val_mask_2d

class MultiRegionDataset(Dataset):
    def __init__(self, regions, batch_size=32, mode='train'):
        self.regions = regions
        self.batch_size = batch_size
        self.mode = mode
        
        self.datasets = {}
        for region in regions: 
            self.datasets[region] = ODDataset(mode=mode, region_name=region)
            
        # 서울 아닌 지역은 static feature 0으로 설정
        for r in regions:
            if r != 'seoul':
                self.datasets[r].X_cont = np.zeros((self.datasets[r].num_dongs, self.datasets['seoul'].X_cont.shape[1]), dtype=np.float32)
                self.datasets[r].X_prop_multi = np.zeros((self.datasets[r].num_dongs, self.datasets['seoul'].X_prop_multi.shape[1]), dtype=np.float32)
                self.datasets[r].X_prop_single = np.zeros((self.datasets[r].num_dongs, self.datasets['seoul'].X_prop_single.shape[1]), dtype=np.float32)
                self.datasets[r].X_zero = np.zeros((self.datasets[r].num_dongs, self.datasets['seoul'].X_zero.shape[1]), dtype=np.float32)
                self.datasets[r].pop_raw = np.zeros((self.datasets[r].num_dongs,), dtype=np.float32)
                
        self.length = 1000 if mode == 'train' else 1

    def __len__(self):
        return self.length

    # 마스크 사이즈 설정
    @property
    def max_mask_size(self) -> int:
        return getattr(self, '_max_mask_size', 1)

    @max_mask_size.setter
    def max_mask_size(self, value: int)-> None:
        self._max_mask_size = value
        for ds in self.datasets.values():
            ds.max_mask_size = value
            
    @property
    def mask_cont_indices(self):
        return self.datasets['seoul'].mask_cont_indices
        
    def __getitem__(self, idx) -> dict:
        '''
            return:
                - static features: (B, N, F) -
                x_cont: continuous static features
                x_prop_multi: multi-class proportion static features
                x_prop_single: single-class proportion static features
                x_zero: zero-valued static features
                
                - dynamic features: (B, N, N) -
                x_dist: distance matrix (log-scaled)
                x_od_masked: OD matrix with masked nodes
                y_od: ground truth OD matrix
                
                mask: (B, N) - boolean mask where True means masked (predict this)
                has_static: (B,) - boolean indicating if the region has static features (True for Seoul, False for others)
                
        '''
        
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
            
        # ODDataset
        dataset = self.datasets[region]
        
        b_x_cont = []
        b_x_prop_multi = []
        b_x_prop_single = []
        b_x_zero = []
        b_x_dist = []
        b_x_od_masked = []
        b_y_od = []
        b_mask = []
        b_loss_mask = []
        b_pop_raw = []
        b_has_static = []
        
        B = self.batch_size if self.mode == 'train' else 1
        
        for _ in range(B):
            item = dataset.__getitem__(idx)
            b_x_cont.append(item['X_cont'])
            b_x_prop_multi.append(item['X_prop_multi'])
            b_x_prop_single.append(item['X_prop_single'])
            b_x_zero.append(item['X_zero'])
            b_x_dist.append(item['X_dist'])
            b_x_od_masked.append(item['X_OD_masked'])
            b_y_od.append(item['y_OD'])
            b_mask.append(item['mask'])
            b_loss_mask.append(item.get('loss_mask', item['mask']))
            b_pop_raw.append(item['pop_raw'])
            b_has_static.append(torch.tensor(region == 'seoul', dtype=torch.bool))
            
        return {
            'X_cont': torch.stack(b_x_cont),
            'X_prop_multi': torch.stack(b_x_prop_multi),
            'X_prop_single': torch.stack(b_x_prop_single),
            'X_zero': torch.stack(b_x_zero),
            'X_dist': torch.stack(b_x_dist),
            'X_OD_masked': torch.stack(b_x_od_masked),
            'y_OD': torch.stack(b_y_od),
            'mask': torch.stack(b_mask),
            'loss_mask': torch.stack(b_loss_mask),
            'pop_raw': torch.stack(b_pop_raw),
            'has_static': torch.stack(b_has_static)
        }