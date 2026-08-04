import sys
import torch
import os

sys.path.append('src')
sys.path.append('src/mae-year')

from dataset import ODDataset
from evaluation.fixed_eval_utils import make_base_data

dataset = ODDataset(year='2023')
base_mem = make_base_data(dataset)

base_pt = torch.load('dataset/fixed_eval/base_data_2023.pt', map_location='cpu', weights_only=False)

for k in base_mem.keys():
    mem = torch.tensor(base_mem[k])
    pt = torch.tensor(base_pt[k])
    if not torch.allclose(mem, pt, equal_nan=True):
        print(f"DIFFERENCE in {k}")
        print("MEM:", mem.max().item(), mem.min().item())
        print("PT:", pt.max().item(), pt.min().item())
    else:
        print(f"{k} matches.")
