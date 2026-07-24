import torch
from src.config import VAL_CITIES_CODES, TEST_CITIES_CODES
import pandas as pd
from src.config import DONG_CODE_PATH
import numpy as np

dong_df = pd.read_excel(DONG_CODE_PATH)
dongs = dong_df['dong_code'].astype(int).values
dong2idx_map = {code: i for i, code in enumerate(dongs)}

test_codes = []
for k,v in TEST_CITIES_CODES.items():
    test_codes.extend([int(c) for c in v])
test_idx = [dong2idx_map[c] for c in test_codes if c in dong2idx_map]

val_data = torch.load('dataset/fixed_eval/fixed_val_dataset.pt')

for city in val_data:
    for task in val_data[city]:
        sample = val_data[city][task][0]
        od_sum = sample['X_OD_masked'][test_idx, :].sum().item()
        static_sum = sample['X_static'][test_idx, :-2].sum().item()
        is_masked_sum = sample['X_static'][test_idx, -2].sum().item()
        print(f"City {city} Task {task}: OD_sum={od_sum}, static_sum={static_sum}, is_masked_sum={is_masked_sum}, len_test={len(test_idx)}")
