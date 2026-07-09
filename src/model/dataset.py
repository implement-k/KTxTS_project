import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_CITIES_CODES, TRAIN_CONFIG, DONG_CODE_PATH

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class ODDataset(Dataset):
    def __init__(self, data_dir, mode='train'):
        self.data_dir = data_dir
        self.mode = mode
        self.max_mask_size = TRAIN_CONFIG['max_mask_size']
        
        # Load distance and OD matrix (only for the 1138 Capital nodes)
        self.X_OD = np.load(os.path.join(data_dir, 'X_OD_2D_cap.npy')) # 1138 x 1138
        self.X_dist = np.load(os.path.join(data_dir, 'X_distance_cap.npy')) # 1138 x 1138
        
        # CSV에서 Static Feature 로드 (비수치형 컬럼 제외, 1138행)
        csv_path = os.path.join(data_dir, 'Final_Static_Features_cap.csv')
        static_df = pd.read_csv(csv_path)
        raw_static = static_df.drop(columns=['dong_code', 'sigungu_code', 'OD행정동코드', '행정동명'], errors='ignore').values
        
        # 피처 정규화 (StandardScaler) - NaN 방지
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        self.X_static = scaler.fit_transform(raw_static)
        
        # 마스킹 여부를 알려주는 Indicator 컬럼 추가 (0.0으로 초기화)
        indicator = np.zeros((self.X_static.shape[0], 1), dtype=np.float32)
        self.X_static = np.concatenate([self.X_static, indicator], axis=1)
        
        self.num_nodes = self.X_OD.shape[0]
            
        # 선택한 도시의 인덱스 찾기
        test_city_indices = find_dong_indices(DONG_CODE_PATH)
        
        # Test 도시의 종사자 수(인덱스 3)와 사업체 수(인덱스 4) 결측 처리 (0으로 마스킹)
        # target leakage 방지를 위해 예측 대상 도시의 실측 데이터는 모델에 제공하지 않습니다.
        self.test_indices = np.array(test_city_indices)
        self.X_static[self.test_indices, 3] = 0.0
        self.X_static[self.test_indices, 4] = 0.0
        self.X_static[self.test_indices, -1] = 1.0 # is_masked = 1
        
        # train test 분리
        self.all_indices = np.arange(self.num_nodes)
        self.train_indices = np.setdiff1d(self.all_indices, self.test_indices)
        
        # 정규화
        self.X_dist = np.log1p(self.X_dist)
        self.X_OD = np.log1p(self.X_OD)
        
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
        
        # Target OD
        y_OD = self.X_OD.copy()
        
        # Masked OD
        X_OD_masked = self.X_OD.copy()
        X_OD_masked[mask, :] = 0
        X_OD_masked[:, mask] = 0
        
        # Static Feature Dynamic Masking
        X_static_masked = self.X_static.copy()
        X_static_masked[mask, 3] = 0.0 # worker_count
        X_static_masked[mask, 4] = 0.0 # business_count
        X_static_masked[mask, -1] = 1.0 # is_masked indicator
        
        return {
            'X_static': torch.tensor(X_static_masked, dtype=torch.float32),
            'X_dist': torch.tensor(self.X_dist, dtype=torch.float32),
            'X_OD_masked': torch.tensor(X_OD_masked, dtype=torch.float32),
            'y_OD': torch.tensor(y_OD, dtype=torch.float32),
            'mask': torch.tensor(mask, dtype=torch.bool)
        }

def find_dong_indices(dong_file):
    dong_code = pd.read_excel(dong_file)['읍면동'].astype(str).values
    idx_map = {code: i for i, code in enumerate(dong_code)}
    
    test_city_indices = []
    for _, codes in TEST_CITIES_CODES.items():
        for code in codes:
            if code in idx_map:
                test_city_indices.append(idx_map[code])
                
    del dong_code, idx_map # 메모리 관리
    return np.array(test_city_indices)