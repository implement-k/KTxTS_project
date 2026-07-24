import torch
import torch.nn as nn

B, N, D, nhead = 2, 5, 16, 4
x = torch.rand(B, N, D)
bias = torch.rand(B, N, N, nhead)

layer = nn.TransformerEncoderLayer(d_model=D, nhead=nhead, batch_first=True)
encoder = nn.TransformerEncoder(layer, num_layers=1)

try:
    out = encoder(x, mask=bias)
    print("Success")
except Exception as e:
    print(f"Error: {e}")
