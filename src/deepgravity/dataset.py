import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MASKING_COLUMNS,
    TEST_CITIES_19_CODES, TEST_CITIES_23_CODES, 
    VAL_CITIES_19_CODES, VAL_CITIES_23_CODES, 
    DONG_CODE_19_PATH, DONG_CODE_23_PATH,
    DIST_DATA_19_PATH, DIST_DATA_23_PATH, 
    STATIC_DATA_19_PATH, STATIC_DATA_23_PATH, 
    OD_DATA_19_PATH, OD_DATA_23_PATH
)

'''
    이 코드는 수정할 필요 없을거야. models.py에서 코드 수정하면 돼.
'''


class ODDataset:
    def __init__(self, year='2023', imputation='zero', use_raw_static=False):      
        if year == '2019':
            DONG_CODE_PATH = DONG_CODE_19_PATH
            DIST_DATA_PATH = DIST_DATA_19_PATH
            STATIC_DATA_PATH = STATIC_DATA_19_PATH
            OD_DATA_PATH = OD_DATA_19_PATH
            TEST_CITIES_CODES = TEST_CITIES_19_CODES
            VAL_CITIES_CODES = VAL_CITIES_19_CODES
        elif year == '2023':
            DONG_CODE_PATH = DONG_CODE_23_PATH
            DIST_DATA_PATH = DIST_DATA_23_PATH
            STATIC_DATA_PATH = STATIC_DATA_23_PATH
            OD_DATA_PATH = OD_DATA_23_PATH
            TEST_CITIES_CODES = TEST_CITIES_23_CODES
            VAL_CITIES_CODES = VAL_CITIES_23_CODES

        # === 행정동 코드 로드 ===
        dong_df = pd.read_excel(DONG_CODE_PATH)
        dongs = dong_df['dong_code'].astype(int).values
        self.num_nodes = len(dongs)   # 전체 동 개수
        self.dong_codes = dongs
        dong2idx_map = {code: i for i, code in enumerate(dongs)}
        
        # === OD 매트릭스 로드 (N,N) ===
        self.X_OD = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        od_path = os.path.join(os.path.dirname(OD_DATA_PATH), f'od_data_{year}.csv')
        if not os.path.exists(od_path):
            od_path = OD_DATA_PATH
        od_df = pd.read_csv(od_path)
        
        # OD 데이터에서 유효한 행정동만 필터링
        o_indices = od_df['O_dong_code'].map(dong2idx_map).values
        d_indices = od_df['D_dong_code'].map(dong2idx_map).values
        valid_mask = pd.notna(o_indices) & pd.notna(d_indices)
        
        o_idx_valid = np.asarray(o_indices[valid_mask].astype(int))
        d_idx_valid = np.asarray(d_indices[valid_mask].astype(int))
        
        # (N, N)으로 합산
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
        
        # 거리 매트릭스에 값 채우기 (log1p 변환하여 스케일 안정화)
        raw_distances = np.asarray(dist_df['distance'].values[dist_mask])
        self.X_dist[o_dist[dist_mask], d_dist[dist_mask]] = np.log1p(raw_distances)
        
        # === Static Feature 로드 ===
        static_path = os.path.join(os.path.dirname(STATIC_DATA_PATH), f'final_static_features_{year}.csv')
        if not os.path.exists(static_path):
            static_path = STATIC_DATA_PATH
        static_df = pd.read_csv(static_path)
        # Rename 2023 specific columns to standard names
        col_mapping = {}
        for c in static_df.columns:
            if c.startswith('station_count_2023_'):
                col_mapping[c] = c.replace('station_count_2023_', 'station_count_')
        static_df.rename(columns=col_mapping, inplace=True)
        
        static_df['dong_code'] = static_df['dong_code'].astype(int)
        
        # 행정동 코드 기준으로 결측치 0으로 채우기
        static_df = static_df.set_index('dong_code').reindex(dongs).reset_index()
        static_df.fillna(0, inplace=True)
        
        # Ensure station_density_지하철 exists in 2023
        if 'station_density_지하철' not in static_df.columns:
            static_df['station_density_지하철'] = static_df['station_count_지하철'] / (static_df['행정동전체면적_m2'] + 1e-5)
        
        feature_cols = [c for c in static_df.columns if c not in ['dong_code', 'dong_name', '시군구']]
        feature_cols = sorted(feature_cols)
        raw_static = static_df[feature_cols].values
        self.masking_indices = [feature_cols.index(c) for c in MASKING_COLUMNS if c in feature_cols]
        
        # === 선택한 도시의 인덱스 찾기 및 train/val/test 분리 ===
        self.test_indices = self._find_dong_indices(dong2idx_map, TEST_CITIES_CODES)
        self.val_indices = self._find_dong_indices(dong2idx_map, VAL_CITIES_CODES)
        self.all_indices = np.arange(self.num_nodes)
        
        self.val_city_indices = {
            city: np.array([dong2idx_map[int(c)] for c in codes if int(c) in dong2idx_map])
            for city, codes in VAL_CITIES_CODES.items()
        }
        
        # Test와 Val을 제외한 나머지를 Train으로 설정
        exclude_indices = np.union1d(self.test_indices, self.val_indices)
        self.train_indices = np.setdiff1d(self.all_indices, exclude_indices)

        # 피처 정규화
        scaler = StandardScaler()
        scaler.fit(raw_static[self.train_indices])
        self.X_static = scaler.transform(raw_static)
        self.X_static_raw = raw_static.copy()
        
        # 학습 데이터의 컬럼별 평균 계산 (mean imputation용)
        train_means = np.mean(self.X_static[self.train_indices], axis=0)
        train_means_raw = np.mean(self.X_static_raw[self.train_indices], axis=0)
        
        for m_idx in self.masking_indices:
            if imputation == 'mean':
                self.X_static[exclude_indices, m_idx] = train_means[m_idx]
                self.X_static_raw[exclude_indices, m_idx] = train_means_raw[m_idx]
            else: # default: zero
                self.X_static[exclude_indices, m_idx] = 0.0
                self.X_static_raw[exclude_indices, m_idx] = 0.0
        
        # train_mask 생성
        self.train_mask = np.zeros(self.num_nodes, dtype=bool)
        self.train_mask[self.train_indices] = True
        
        # 정답 총유출량, 총발생량 예측 (전체 실제 총량 사용)
        self.y_o = np.sum(self.X_OD, axis=1)
        self.y_d = np.sum(self.X_OD, axis=0)
        
        self.y_o_val = np.sum(self.X_OD, axis=1)   
        self.y_d_val = np.sum(self.X_OD, axis=0)
        
        # train 마스크
        self.X_static_train = self.X_static_raw[self.train_mask] if use_raw_static else self.X_static[self.train_mask] 
        
        print(f"I: Dataset 초기화 완료 (Imputation: {imputation})")
        
    def _find_dong_indices(self, idx_map, cities_codes):
        all_codes = (int(code) for codes in cities_codes.values() for code in codes)
        return np.array([idx_map[c] for c in all_codes if c in idx_map])
