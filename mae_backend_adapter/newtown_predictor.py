import os
import json
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
import torch
import pickle

from mae_backend_adapter import (
    MAEPredictor,
    ModelInputs,
    load_static_scaler
)

class NewtownPreprocessor:
    def __init__(self, newtown_name: str, period: str):
        self.newtown_name = newtown_name
        self.period = period
        self.base_dir = Path(__file__).resolve().parent / "newtown" / newtown_name
        
    def prepare(self) -> ModelInputs:
        if not self.base_dir.exists():
            pytest.skip(f"Newtown data for {self.newtown_name} not found.")
            
        # 1. OD list
        od_df = pd.read_excel(self.base_dir / "OD_dong_list_2023.xlsx")
        codes = od_df['dong_code'].astype(str).tolist()
        N = len(codes)
        
        # 2. Distance
        dist_df_raw = pd.read_csv(self.base_dir / "dong_distance.csv")
        dist_df_raw['O_dong_code'] = dist_df_raw['O_dong_code'].astype(str)
        dist_df_raw['D_dong_code'] = dist_df_raw['D_dong_code'].astype(str)
        dist_df = dist_df_raw.pivot(index='O_dong_code', columns='D_dong_code', values='distance')
        dist_df = dist_df.reindex(index=codes, columns=codes).fillna(0.0)
        
        x_dist = torch.tensor(dist_df.values, dtype=torch.float64)
        
        # 3. Adjacency
        with open(self.base_dir / "dong_adjacency.pkl", "rb") as f:
            adj = pickle.load(f)
            
        a_spatial_np = np.zeros((N, N), dtype=np.float64)
        code_to_idx = {str(code): i for i, code in enumerate(codes)}
        
        if isinstance(adj, dict):
            for node, neighbors in adj.items():
                str_node = str(node)
                if str_node in code_to_idx:
                    idx_a = code_to_idx[str_node]
                    for neighbor in neighbors:
                        str_neighbor = str(neighbor)
                        if str_neighbor in code_to_idx:
                            idx_b = code_to_idx[str_neighbor]
                            a_spatial_np[idx_a, idx_b] = 1.0
                            a_spatial_np[idx_b, idx_a] = 1.0
        else:
           raise ValueError("Adjacency data is not in expected dictionary format.")
            
        a_spatial = torch.tensor(a_spatial_np, dtype=torch.float64)
            
        # 4. Static features
        static_df = pd.read_csv(self.base_dir / f"static_features_{self.period}.csv")
        static_df['dong_code'] = static_df['dong_code'].astype(str)
        static_df = static_df.set_index('dong_code').reindex(codes).reset_index()
        static_df.fillna(0, inplace=True)
        
        feature_names = [col for col in static_df.columns if col not in ['dong_code', 'dong_name']]
        feature_names = sorted(feature_names) # 정렬하여 일관성 유지
        
        # 5. mask 코드 파싱
        is_masked = np.zeros(N, dtype=bool)
        mask_json_path = self.base_dir / "mask_code.json"
        
        newtown_zone_codes = []
        if mask_json_path.exists():
            with open(mask_json_path, "r", encoding="utf-8") as f:
                mask_data = json.load(f)
                mask_city_dict = mask_data.get("mask_city", {})
                
                # dict의 value(코드)들을 추출 (문자열 변환)
                mask_codes = set(str(v) for v in mask_city_dict.values())
                
                self.val_codes = set()
                        
                newtown_zone_codes = list(mask_codes)
                
                for i, code in enumerate(codes):
                    if code in mask_codes:
                        is_masked[i] = True
        else:
            raise FileNotFoundError(f"Mask code JSON file not found at {mask_json_path}")
            
        # od_data_2023.csv 읽기
        od_df_2023 = pd.read_csv(self.base_dir.parent / "od_data_2023.csv")
        od_matrix_df = pd.DataFrame(0.0, index=codes, columns=codes)
        
        o_indices = od_df_2023['O_dong_code'].astype(str).map({c: i for i, c in enumerate(codes)})
        d_indices = od_df_2023['D_dong_code'].astype(str).map({c: i for i, c in enumerate(codes)})
        valid_mask = o_indices.notna() & d_indices.notna()
        
        purposes = ['귀가', '출근', '등교', '업무', '기타']
        calculated_total = od_df_2023[purposes].sum(axis=1)
        
        for idx in od_df_2023[valid_mask].index:
            o_code = str(od_df_2023.loc[idx, 'O_dong_code'])
            d_code = str(od_df_2023.loc[idx, 'D_dong_code'])
            od_matrix_df.loc[o_code, d_code] += calculated_total.loc[idx]
            
        x_od_masked = torch.tensor(od_matrix_df.values, dtype=torch.float64)
        
        # 정규화 (거리 및 통행량 로그 변환) - dataset.py와 동일하게 동작
        x_dist = torch.log1p(x_dist)
        x_od_masked = torch.log1p(x_od_masked)
        
        self.original_x_od = x_od_masked.clone()
        
        # 신도시(mask)에 해당하는 O 또는 D는 0으로 처리 (마스킹)
        mask_tensor = torch.tensor(is_masked, dtype=torch.bool)
        masked_pairs = mask_tensor.unsqueeze(1) | mask_tensor.unsqueeze(0)
        x_od_masked = torch.where(masked_pairs, torch.zeros_like(x_od_masked), x_od_masked)
        
        is_merged = np.zeros(N, dtype=bool)
        
        scaler = load_static_scaler()
        scaled_features = scaler.transform_with_indicators(
            static_df[feature_names].values,
            feature_names=feature_names,
            is_masked=is_masked,
            is_merged=is_merged
        )
        x_static = torch.tensor(scaled_features, dtype=torch.float64)
        
        # 6. Model Inputs
        return ModelInputs(
            x_static=x_static,
            x_od_masked=x_od_masked,
            x_dist=x_dist,
            a_spatial=a_spatial,
            mask=mask_tensor,
            city_codes=codes,
            newtown_zone_codes=newtown_zone_codes, 
            population_allocation_method="default"
        )

@pytest.mark.parametrize("newtown_name", ["all","changneung","gyosan","wangsuk"])
@pytest.mark.parametrize("period", ["initial", "middle", "final"])
def test_real_newtown_smoke(newtown_name, period):
    preprocessor = NewtownPreprocessor(newtown_name=newtown_name, period=period)
    inputs = preprocessor.prepare()
    
    try:
        predictor = MAEPredictor(device="cpu")
    except Exception as e:
        pytest.skip(f"Cannot initialize predictor, possibly missing model: {e}")
    
    # Run predict
    # predict_from_tensors를 호출하면 JSON dict가 반환되므로, 텐서를 직접 검증하기 위해 내부 메서드를 호출합니다.
    matrix, metadata, origins, destinations, zones = predictor._run_full(inputs, request_metadata=None)
    
    assert matrix.shape == (len(inputs.city_codes), len(inputs.city_codes))
    assert not torch.isnan(matrix).any(), "Prediction contains NaN values."
    
    # 0 handling check
    assert matrix.min() >= 0.0, "Prediction must be >= 0."
    print(f"Smoke test for {newtown_name} passed with OD matrix shape {matrix.shape}")
    
