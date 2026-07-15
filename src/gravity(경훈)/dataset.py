import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TEST_CITIES_CODES, TRAIN_CONFIG, DONG_CODE_PATH,
    DIST_DATA_PATH, STATIC_DATA_PATH, OD_DATA_PATH
)

'''
    이 코드는 수정할 필요 없을거야. models.py에서 코드 수정하면 돼.
'''


class ODDataset:
    def __init__(self):
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
        
        # 선택한 도시의 인덱스 찾기 및 train/test 분리
        self.test_indices = self._find_dong_indices(dong2idx_map)
        self.all_indices = np.arange(self.num_nodes)
        self.train_indices = np.setdiff1d(self.all_indices, self.test_indices)
        
        
        feature_cols = [c for c in static_df.columns if c not in ['dong_code', 'dong_name']]
        raw_static = static_df[feature_cols].values
        self.masking_indices = [feature_cols.index(c) for c in MASKING_COLUMNS if c in feature_cols]
        
        # 피처 정규화
        scaler = StandardScaler()
        scaler.fit(raw_static[self.train_indices])
        self.X_static = scaler.transform(raw_static)
        
        for m_idx in self.masking_indices:
            self.X_static[self.test_indices, m_idx] = 0.0
        self.X_static[self.test_indices, -1] = 1.0 # is_masked = 1
        
        print("Dataset 초기화 완료")
        
    def _find_dong_indices(self, idx_map):
        test_city_indices = []
        for _, codes in TEST_CITIES_CODES.items():
            for str_code in codes:
                code_int = int(str_code)
                if code_int in idx_map:
                    test_city_indices.append(idx_map[code_int])
        return np.array(test_city_indices)
        
