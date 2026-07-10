import torch
import numpy as np
y_real = torch.tensor([0.0]*10000 + [100.0]*100 + [10000.0]*10)
y_log = torch.log1p(y_real)

pred_log4 = torch.zeros_like(y_log)
pred_log4.requires_grad = True
optimizer4 = torch.optim.Adam([pred_log4], lr=0.1)
for i in range(100):
    real_target = y_real
    weight = 1.0 + 10.0 * (real_target / 100.0)
    loss = (((pred_log4 - y_log)**2) * weight).mean()
    optimizer4.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_([pred_log4], max_norm=1.0)
    optimizer4.step()

pred_real4 = torch.expm1(pred_log4).detach().numpy()
rmse4 = np.sqrt(np.mean((y_real.numpy() - pred_real4)**2))
print(f"Log-scale weighted (real target weights). RMSE: {rmse4:.2f}")
