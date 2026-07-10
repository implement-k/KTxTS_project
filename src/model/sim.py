import torch
import numpy as np
y_real = torch.tensor([0.0]*10000 + [100.0]*100 + [10000.0]*10)
y_log = torch.log1p(y_real)
pred_log = torch.zeros_like(y_log)
pred_log.requires_grad = True
optimizer = torch.optim.Adam([pred_log], lr=0.1)

for i in range(100):
    weight = 1.0 + 10.0 * y_log
    loss = (((pred_log - y_log)**2) * weight).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

pred_real = torch.expm1(pred_log).detach().numpy()
rmse = np.sqrt(np.mean((y_real.numpy() - pred_real)**2))
print(f"Log-scale weighted MSE. RMSE: {rmse:.2f}")

pred_log2 = torch.zeros_like(y_log)
pred_log2.requires_grad = True
optimizer2 = torch.optim.Adam([pred_log2], lr=0.1)
for i in range(100):
    real_pred = torch.expm1(pred_log2)
    # real MSE
    loss = ((real_pred - y_real)**2).mean()
    optimizer2.zero_grad()
    loss.backward()
    optimizer2.step()
pred_real2 = torch.expm1(pred_log2).detach().numpy()
rmse2 = np.sqrt(np.mean((y_real.numpy() - pred_real2)**2))
print(f"Real-scale MSE directly. RMSE: {rmse2:.2f}")

