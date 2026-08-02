import os
print("Starting eval_hybrid.py...")
import sys
import torch
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, '../src'))
sys.path.insert(0, os.path.join(current_dir, '../src/mae-year'))
from models import ODMAE
from dataset import ODDataset
from validation import evaluate_and_report, summarize_results
from evaluation.fixed_eval_utils import make_base_data

def eval_year(year, model_path):
    print(f"\n[Evaluating {year} with {model_path}]")
    
    st = torch.load(model_path, map_location='cpu')
    if 'model_state_dict' in st:
        st = st['model_state_dict']
        
    d_model = st['feature_embed.0.weight'].shape[0]
    num_features = st['feature_embed.0.weight'].shape[1]
    has_mask = 'mask_proj.weight' in st
    nhead = st['distance_bias.weight'].shape[1] if 'distance_bias.weight' in st else 4
    has_self_loop = 'self_loop_predictor.0.weight' in st
    
    model = ODMAE(num_features=num_features, d_model=d_model, nhead=nhead, num_layers=4,
                  od_embed_layers=2, use_distance_friction=False, use_self_loop_predictor=has_self_loop,
                  use_mask_channel=has_mask)
    model.load_state_dict(st, strict=False)
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device)

    dataset = ODDataset(year=year)
    base_data = make_base_data(dataset)
    val_meta = torch.load(os.path.join(current_dir, f'../dataset/fixed_eval/fixed_val_meta_{year}.pt'), map_location='cpu', weights_only=False)

    records = evaluate_and_report(
        base_data=base_data,
        val_meta=val_meta,
        model=model,
        year_label=year,
        split_name='val',
        n_workers=8,
        device=device,
        use_lgbm_self_loop=True
    )
    
    summarize_results(records, ['year', 'task'], f"[{year}] Task Summary")
    summarize_results(records, ['year'], f"[{year}] Overall Summary")

if __name__ == "__main__":
    model_path = os.path.join(current_dir, '../best_model/mae:hybrid-86epoch.pth')
    eval_year('2019', model_path)
    eval_year('2023', model_path)
