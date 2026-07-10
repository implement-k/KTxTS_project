import torch
import numpy as np
y_real = torch.tensor([0.0]*10000 + [100.0]*100 + [10000.0]*10)
y_log = torch.log1p(y_real)

pred_log3 = torch.zeros_like(y_log)
pred_log3.requires_grad = True
optimizer3 = torch.optim.Adam([pred_log3], lr=0.1)
for i in range(100):
    pred_real = torch.expm1(pred_log3)
    target_real = y_real
    loss = torch.nn.functional.huber_loss(pred_real, target_real, delta=500.0)
    optimizer3.zero_grad()
    loss.backward()
    # clip grad
    torch.nn.utils.clip_grad_norm_([pred_log3], max_norm=1.0)
    optimizer3.step()

pred_real3 = torch.expm1(pred_log3).detach().numpy()
rmse3 = np.sqrt(np.mean((y_real.numpy() - pred_real3)**2))
print(f"Real-scale Huber Loss. RMSE: {rmse3:.2f}")
