import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'mae-old'))
from models import SpatialODMAE

model = SpatialODMAE(num_nodes=1137, num_features=20, d_model=128, num_layers=4, nhead=8, loss_type='hybrid_od')
torch.save(model.state_dict(), 'dummy_ckpt.pt')
