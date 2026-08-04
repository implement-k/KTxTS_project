import os
import sys
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.mae_year.dataset import ODDataset
from src.evaluation.fixed_eval_utils import make_base_data
from src.mae_year.validation import evaluate_and_report
from src.mae_year.models import ODMAE

device = 'cpu'
year = '2023'
model_path = 'best_model/mae:hybrid_cpc-2023.pth'

# 1. Load base_data.pt
fixed_eval_dir = 'dataset/fixed_eval'
base_data_pt = torch.load(os.path.join(fixed_eval_dir, f'base_data_{year}.pt'), map_location='cpu', weights_only=False)
meta_data = torch.load(os.path.join(fixed_eval_dir, f'fixed_val_meta_{year}.pt'), map_location='cpu', weights_only=False)

# 2. make_base_data
dataset = ODDataset(year=year)
base_data_make = make_base_data(dataset)

# 3. Model
F = dataset.X_static.shape[1]
for use_self_loop in [True, False]:
    print(f"\n--- Testing with use_self_loop_predictor={use_self_loop} ---")
    model = ODMAE(num_features=F, use_self_loop_predictor=use_self_loop).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True), strict=False)
        model.eval()
        
        records_pt = evaluate_and_report(base_data_pt, meta_data, model, year, 'val', 1, device)
        cpc_pt = np.mean([r['cpc'] for r in records_pt if r is not None])
        print(f"base_data.pt CPC: {cpc_pt:.4f}")
        
        records_make = evaluate_and_report(base_data_make, meta_data, model, year, 'val', 1, device)
        cpc_make = np.mean([r['cpc'] for r in records_make if r is not None])
        print(f"make_base_data CPC: {cpc_make:.4f}")
        
        # 4. make_base_data but force test_indices = []
        base_data_make_leak = base_data_make.copy()
        base_data_make_leak['test_indices'] = []
        records_make_leak = evaluate_and_report(base_data_make_leak, meta_data, model, year, 'val', 1, device)
        cpc_make_leak = np.mean([r['cpc'] for r in records_make_leak if r is not None])
        print(f"make_base_data (unmasked test_indices) CPC: {cpc_make_leak:.4f}")
        
    except Exception as e:
        print("Error:", e)
