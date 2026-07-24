import os
import re

def update():
    with open('src/dataset.py', 'r') as f:
        content = f.read()
        
    # We want to replace __init__ of ODDataset
    # and add MultiRegionDataset at the end
    
    # Let's find ODDataset class
    # and replace it with a new implementation that supports region_name.
    
    new_od_dataset = """class ODDataset(Dataset):
    def __init__(self, mode='train', use_nan_masking=False, use_log_transform=True, use_od=True, predict_only_masked=False, use_residual=False, region_name='seoul'):
        self.mode = mode
        self.region_name = region_name
        self.max_mask_size = TRAIN_CONFIG['min_mask_size']
        
        # twostage용 코드
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
        od_path = os.path.join(DATA_DIR, f'od_{region}.csv')
        od_df = pd.read_csv(od_path)
        
        # 존 ID 리스트 추출
        zones = sorted(list(set(od_df['출발'].unique()) | set(od_df['도착'].unique())))
        self.num_nodes = len(zones)
        zone2idx = {z: i for i, z in enumerate(zones)}
        
        self.X_OD = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        o_idx = od_df['출발'].map(zone2idx).values
        d_idx = od_df['도착'].map(zone2idx).values
        purposes = ['귀가', '출근', '등교', '업무', '기타']
        # 엑셀마다 목적 컬럼이 조금씩 다를 수 있으므로 있는 컬럼만
        available_purposes = [c for c in purposes if c in od_df.columns]
        if len(available_purposes) == 0:
            if '합계' in od_df.columns:
                calculated_total = od_df['합계']
            else:
                calculated_total = od_df.iloc[:, 2:].sum(axis=1) # 임시 방편
        else:
            calculated_total = od_df[available_purposes].sum(axis=1)
        
        self.X_OD[o_idx, d_idx] = calculated_total.values
        
        # 2. Distance 매트릭스 로드
        dist_path = os.path.join(DATA_DIR, 'processed', f'{region}_dist.csv')
        dist_df = pd.read_csv(dist_path, index_col=0)
        # DataFrame의 인덱스와 컬럼이 존 ID라고 가정
        self.X_dist = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        
        for i, z1 in enumerate(zones):
            for j, z2 in enumerate(zones):
                if str(z1) in dist_df.index.astype(str) and str(z2) in dist_df.columns.astype(str):
                    self.X_dist[i, j] = dist_df.loc[z1, str(z2)] if str(z2) in dist_df.columns else dist_df.loc[str(z1), str(z2)]
        
        # 3. Static Features 로드 (타 지역은 데이터가 없으므로 0으로 패딩)
        # seoul 기준 static 피처 갯수를 맞춰준다 (worker_count 등).
        # MASKING_COLUMNS 수 + indicator 1개 등을 포함해서 동일하게 맞춰야 함.
        # 서울 데이터의 feature_cols 길이는 보통 10개 내외임. 
        # 정확히 맞추기 위해 서울 스태틱을 잠깐 로드해서 형태 파악
        seoul_static = pd.read_csv(STATIC_DATA_PATH)
        feature_cols = [c for c in seoul_static.columns if c not in ['dong_code', 'dong_name'] and not c.startswith('station_density') and c not in ['worker_density', 'business_density']]
        # 밀도 파생변수 포함
        F = len(feature_cols) + 3 # worker_density, business_density, station_density
        
        self.masking_indices = [0, 1] # 대략적으로 처음 2개가 worker, business라고 가정
        # 0.0 으로 패딩, 마지막 컬럼 indicator 포함
        self.X_static = np.zeros((self.num_nodes, F + 1), dtype=np.float32)
        
        self.test_indices = np.array([], dtype=int)
        self.all_indices = np.arange(self.num_nodes)
        self.train_indices = self.all_indices
        
        if use_log_transform:
            self.X_dist = np.log1p(self.X_dist)
            self.X_OD = np.log1p(self.X_OD)
            
        print(f"[{region}] Dataset 초기화 완료 (N={self.num_nodes})")

    def _init_seoul(self, use_nan_masking, use_log_transform):
        # 행정동 코드 로드
        dong_df = pd.read_excel(DONG_CODE_PATH)
        dongs = dong_df['dong_code'].astype(int).values
        self.num_nodes = len(dongs)   # 전체 동 개수
        dong2idx_map = {code: i for i, code in enumerate(dongs)}
        
        # === OD 매트릭스 로드 (N,N) ===
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
        
        # === 거리 매트릭스 로드 (N, N) ===
        self.X_dist = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        dist_df = pd.read_csv(DIST_DATA_PATH)
        
        o_dist = dist_df['O_dong_code'].map(dong2idx_map).values
        d_dist = dist_df['D_dong_code'].map(dong2idx_map).values
        dist_mask = pd.notna(o_dist) & pd.notna(d_dist)
        
        self.X_dist[o_dist[dist_mask].astype(int), d_dist[dist_mask].astype(int)] = dist_df['distance'].values[dist_mask]
        
        # === Static Feature 로드 ===
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
"""

    pattern = re.compile(r"class ODDataset\(Dataset\):.*?def _find_dong_indices", re.DOTALL)
    new_content = pattern.sub(new_od_dataset + "\n\n    def _find_dong_indices", content)
    
    multi_region = """
class MultiRegionDataset(Dataset):
    def __init__(self, regions, batch_size=32, mode='train', use_log_transform=True):
        self.regions = regions
        self.batch_size = batch_size
        self.mode = mode
        
        self.datasets = {}
        for r in regions:
            self.datasets[r] = ODDataset(mode=mode, use_log_transform=use_log_transform, region_name=r)
            
        # 모든 데이터셋의 static 피처 차원(F)이 동일하도록 seoul 데이터셋의 모양을 기준으로 맞춥니다.
        F = self.datasets['seoul'].X_static.shape[1]
        for r in regions:
            if r != 'seoul':
                self.datasets[r].X_static = np.zeros((self.datasets[r].num_nodes, F), dtype=np.float32)
                
        # 한 에폭 당 이터레이션 수 (배치 사이즈를 내부적으로 처리하므로, 배치 사이즈에 맞춰 길이 조정)
        self.length = 1000 if mode == 'train' else 1

    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        # 1. 랜덤 지역 선택
        if self.mode == 'train':
            # seoul을 메인으로 하되, 타 지역도 섞이게
            region = np.random.choice(self.regions)
        else:
            region = 'seoul'
            
        ds = self.datasets[region]
        
        # 2. 배치 생성 (B, N, F 등)
        b_x_static = []
        b_x_dist = []
        b_x_od_masked = []
        b_y_od = []
        b_mask = []
        
        B = self.batch_size if self.mode == 'train' else 1
        
        for _ in range(B):
            item = ds.__getitem__(idx)
            b_x_static.append(item['X_static'])
            b_x_dist.append(item['X_dist'])
            b_x_od_masked.append(item['X_OD_masked'])
            b_y_od.append(item['y_OD'])
            b_mask.append(item['mask'])
            
        return {
            'X_static': torch.stack(b_x_static),
            'X_dist': torch.stack(b_x_dist),
            'X_OD_masked': torch.stack(b_x_od_masked),
            'y_OD': torch.stack(b_y_od),
            'mask': torch.stack(b_mask)
        }
"""
    new_content += multi_region
    
    with open('src/dataset.py', 'w') as f:
        f.write(new_content)
        
update()
