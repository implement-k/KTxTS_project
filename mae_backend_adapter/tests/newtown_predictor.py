import os
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
import torch
import pickle

from mae_backend_adapter import (
    MAEPredictor,
    MAEProvider,
    ModelInputs,
    load_static_scaler
)

class RealNewtownPreprocessor:
    def __init__(self, newtown_name: str):
        self.newtown_name = newtown_name
        self.base_dir = Path(__file__).resolve().parent.parent / "newtown" / newtown_name
        
    def prepare(self) -> ModelInputs:
        if not self.base_dir.exists():
            pytest.skip(f"Newtown data for {self.newtown_name} not found.")
            
        # 1. OD list
        od_df = pd.read_excel(self.base_dir / "OD_dong_list_2023.xlsx")
        # Ensure string type
        codes = od_df['dong_code'].astype(str).tolist()
        N = len(codes)
        
        # 2. Distance
        dist_df = pd.read_csv(self.base_dir / "dong_distance.csv", index_col=0)
        dist_df.index = dist_df.index.astype(str)
        dist_df.columns = dist_df.columns.astype(str)
        dist_df = dist_df.loc[codes, codes]
        x_dist = torch.tensor(dist_df.values, dtype=torch.float64)
        
        # 3. Adjacency
        with open(self.base_dir / "dong_adjacency.pkl", "rb") as f:
            adj = pickle.load(f)
        if hasattr(adj, 'loc'):
            adj.index = adj.index.astype(str)
            adj.columns = adj.columns.astype(str)
            adj = adj.loc[codes, codes]
            a_spatial = torch.tensor(adj.values, dtype=torch.float64)
        else:
            # If it's just a numpy array, assume order is correct
            a_spatial = torch.tensor(adj, dtype=torch.float64)
            
        # 4. Static features
        static_df = pd.read_csv(self.base_dir / f"newtown_static_features_{self.newtown_name}.csv")
        static_df['dong_code'] = static_df['dong_code'].astype(str)
        static_df = static_df.set_index('dong_code').loc[codes].reset_index()
        
        feature_names = [col for col in static_df.columns if col not in ['dong_code', 'dong_name']]
        
        # Assume mask if dong_name ends with newtown indicators
        # Just simple masking for smoke test
        is_masked = np.zeros(N, dtype=bool)
        is_masked[-5:] = True # Mock last 5 as masked for testing
        is_merged = np.zeros(N, dtype=bool)
        
        scaler = load_static_scaler()
        scaled_features = scaler.transform_with_indicators(
            static_df[feature_names].values,
            feature_names=feature_names,
            is_masked=is_masked,
            is_merged=is_merged
        )
        x_static = torch.tensor(scaled_features, dtype=torch.float64)
        
        # 5. X_OD_masked (just mock with distance-based OD for smoke test)
        # Or all zeros
        x_od_masked = torch.zeros((N, N), dtype=torch.float64)
        
        # 6. Model Inputs
        return ModelInputs(
            x_static=x_static,
            x_od_masked=x_od_masked,
            x_dist=x_dist,
            a_spatial=a_spatial,
            mask=torch.tensor(is_masked, dtype=torch.bool),
            origin_codes=codes,
            destination_codes=codes,
            newtown_zone_codes=[codes[-1]], 
            population_allocation_method="default"
        )

@pytest.mark.parametrize("newtown_name", ["gyosan", "changneung"])
def test_real_newtown_smoke(newtown_name):
    # Setup Provider
    provider = MAEProvider() # Uses default bundled mae.pth
    
    try:
        predictor = MAEPredictor(provider)
    except Exception as e:
        pytest.skip(f"Cannot initialize predictor, possibly missing model: {e}")
        
    preprocessor = RealNewtownPreprocessor(newtown_name=newtown_name)
    inputs = preprocessor.prepare()
    
    # Run predict
    out_adapter = predictor.predict(inputs)
    
    predicted_tensor = out_adapter.as_tensor()
    assert predicted_tensor.shape == (len(inputs.origin_codes), len(inputs.destination_codes))
    assert not torch.isnan(predicted_tensor).any(), "Prediction contains NaN values."
    
    # 0 handling check
    assert predicted_tensor.min() >= 0.0, "Prediction must be >= 0."
    print(f"Smoke test for {newtown_name} passed with OD matrix shape {predicted_tensor.shape}")
