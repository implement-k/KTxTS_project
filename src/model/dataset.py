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
    def __init__(self, mode='train', channel=1, isLogScale=True, protocol='legacy',
                 val_fraction=0.10, val_seed=42):
        """
        protocol: 'legacy'(기존 동작, 하위 호환 100%) 또는 'strict'.
            strict일 때는 mode='val'을 쓸 수 있음 — test_indices를 전혀 건드리지 않고
            train_indices 내부에서 행정구역(시/군/구) group 단위로 떼어낸
            strict_val_indices를 검증 대상으로 사용한다.
        mode: 'train' | 'val' | 'test'. 'val'은 strict protocol 전용.
        """
        self.mode = mode
        self.channel = channel
        self.protocol = protocol
        self.max_mask_size = TRAIN_CONFIG['min_mask_size']
        
        # 행정동 코드 로드
        dong_df = pd.read_excel(DONG_CODE_PATH)
        dongs = dong_df['dong_code'].astype(int).values
        self.num_nodes = len(dongs)   # 전체 동 개수
        dong2idx_map = {code: i for i, code in enumerate(dongs)}
        
        # === OD 매트릭스 로드 (N,N) or (N, N, 5) ===
        if self.channel == 1:
            self.X_OD = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        else:
            self.X_OD = np.zeros((self.num_nodes, self.num_nodes, 5), dtype=np.float32)
        od_df = pd.read_csv(OD_DATA_PATH)
        
        # OD 데이터에서 유효한 행정동만 필터링
        o_indices = od_df['O_dong_code'].map(dong2idx_map).values
        d_indices = od_df['D_dong_code'].map(dong2idx_map).values
        valid_mask = pd.notna(o_indices) & pd.notna(d_indices)
        
        o_idx_valid = o_indices[valid_mask].astype(int)
        d_idx_valid = d_indices[valid_mask].astype(int)
        
        purposes = ['귀가', '출근', '등교', '업무', '기타']
        # mae5 이외의 경우 (N, N)으로 합산
        if self.channel == 1:
            calculated_total = od_df[purposes].sum(axis=1)
            self.X_OD[o_idx_valid, d_idx_valid] = calculated_total.values[valid_mask]
        # mae5인 경우 (N, N, 5)로 각 목적별로 저장
        else:
            # 기존 데이터 3차원으로 변환 후 저장
            for c, purpose in enumerate(purposes):
                if purpose in od_df.columns:
                    self.X_OD[o_idx_valid, d_idx_valid, c] = od_df[purpose].values[valid_mask]
        
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

        # === strict protocol을 위한 group(행정구역) 기반 validation node 분리 ===
        # dong_code // 1000 은 한국 행정동 코드 체계상 시/군/구 단위 그룹키와 일치함
        # (예: 11010530 -> 11010 = 종로구). OD pair를 무작위로 나누지 않고
        # 이 그룹 단위로 validation을 떼어내 공간적으로 인접한 정보가 train/val에
        # 동시에 섞이는 것을 최대한 피한다. 그룹만으로 목표 비율을 맞추기 어려우면
        # seed 고정 개별 노드 샘플링으로 폴백한다.
        self.dong_codes = dongs
        self.strict_train_indices, self.strict_val_indices = self._split_train_val_by_group(
            self.train_indices, self.dong_codes, val_fraction=val_fraction, seed=val_seed)

        # === test 노드와 연결된 OD 쌍을 가리는 마스크 (strict 학습 loss에서 배제용) ===
        # (i, j) 중 어느 한쪽이라도 test_indices에 속하면 True(=test와 접촉).
        test_node_flag = np.zeros(self.num_nodes, dtype=bool)
        test_node_flag[self.test_indices] = True
        self.test_touch_od_mask = test_node_flag[:, None] | test_node_flag[None, :]
        self.non_test_od_mask = ~self.test_touch_od_mask

        # strict_val 노드와 연결된 OD 쌍도 동일한 방식으로 표시(strict 학습에서 함께 배제)
        strict_val_flag = np.zeros(self.num_nodes, dtype=bool)
        strict_val_flag[self.strict_val_indices] = True
        self.strict_val_touch_od_mask = strict_val_flag[:, None] | strict_val_flag[None, :]
        # strict protocol의 train 단계 loss에서 배제해야 하는 전체 영역(test ∪ strict_val 접촉)
        self.strict_excluded_od_mask = self.test_touch_od_mask | self.strict_val_touch_od_mask
        self.strict_train_safe_od_mask = ~self.strict_excluded_od_mask

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
        
        # 정규화 (거리 및 통행량 로그 변환)
        if isLogScale:
            self.X_dist = np.log1p(self.X_dist)
            self.X_OD = np.log1p(self.X_OD)

        print("Dataset 초기화 완료")
        
    def _find_dong_indices(self, idx_map):
        test_city_indices = []
        for _, codes in TEST_CITIES_CODES.items():
            for str_code in codes:
                code_int = int(str_code)
                if code_int in idx_map:
                    test_city_indices.append(idx_map[code_int])
        return np.array(test_city_indices)

    @staticmethod
    def _split_train_val_by_group(train_indices, dong_codes, val_fraction=0.10, seed=42):
        """
        train_indices를 시/군/구 그룹(dong_code // 1000) 단위로 묶어 val_fraction 비율만큼
        그룹째로 validation에 배정한다(OD pair를 무작위로 나누지 않기 위함). 그룹 단위로
        목표 비율에 근접하게 맞추기 어려우면 seed 고정 개별 노드 샘플링으로 폴백한다.
        """
        rng = np.random.RandomState(seed)
        groups = dong_codes[train_indices] // 1000
        unique_groups = np.unique(groups)
        target_val_count = max(1, int(round(len(train_indices) * val_fraction)))

        order = rng.permutation(len(unique_groups))
        picked_groups = []
        picked_count = 0
        for gi in order:
            g = unique_groups[gi]
            g_size = int((groups == g).sum())
            if picked_count + g_size <= target_val_count * 1.5:
                picked_groups.append(g)
                picked_count += g_size
            if picked_count >= target_val_count:
                break

        if picked_groups and abs(picked_count - target_val_count) <= max(3, target_val_count * 0.5):
            val_mask = np.isin(groups, picked_groups)
        else:
            # 그룹 단위로 목표 비율을 맞추기 어려운 경우: 개별 노드 단위 폴백(seed 고정)
            val_mask = np.zeros(len(train_indices), dtype=bool)
            val_pos = rng.choice(len(train_indices), size=target_val_count, replace=False)
            val_mask[val_pos] = True

        strict_val_indices = train_indices[val_mask]
        strict_train_indices = train_indices[~val_mask]
        return strict_train_indices, strict_val_indices

    def __len__(self):
        # train 모드에서는 한 epoch당 1000번, val/test 모드에선 1개의 샘플만 반환
        return 1000 if self.mode == 'train' else 1

    def __getitem__(self, idx):
        if self.mode == 'train':
            # strict protocol이면 strict_val_indices를 완전히 제외한 풀에서만 샘플링
            sample_pool = self.strict_train_indices if self.protocol == 'strict' else self.train_indices

            # 마스크 사이즈 랜덤 선택
            k = np.random.randint(1, self.max_mask_size + 1)

            # 풀 안에서 한개 선택 후 그 동과의 거리 리스트 추출
            dist_list = self.X_dist[np.random.choice(sample_pool)]
            # 그중 풀 밖(= legacy면 test동, strict면 test동+strict_val동) 노드는 제외
            valid_distances = dist_list[sample_pool]
            closest_k_indices = sample_pool[np.argsort(valid_distances)[:k]]

            mask_indices = closest_k_indices
        elif self.mode == 'val':
            # strict protocol 전용: train_indices 내부에서 떼어낸 검증 노드만 사용, test_indices는 전혀 사용하지 않음
            mask_indices = self.strict_val_indices
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

        # mae1인 경우 (N,N)
        if self.channel == 1:
            X_OD_masked[mask, :] = 0
            X_OD_masked[:, mask] = 0
        else:
            X_OD_masked[mask, :, :] = 0
            X_OD_masked[:, mask, :] = 0

        # strict protocol의 train 모드에서는, 커리큘럼상 이번 스텝에 마스킹되지 않은 노드라도
        # test 노드(및 strict_val 노드)와 연결된 실제 OD 값이 모델 입력에 그대로 노출되지 않도록
        # 별도로 0 처리한다(legacy에서는 이 값이 항상 노출되던 것을 strict에서만 차단).
        if self.protocol == 'strict' and self.mode == 'train':
            if self.channel == 1:
                X_OD_masked[self.strict_excluded_od_mask] = 0
            else:
                X_OD_masked[self.strict_excluded_od_mask, :] = 0

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