import numpy as np
import torch

def cpc_score(y_true, y_pred):
    numerator   = 2 * np.sum(np.minimum(y_true, y_pred))
    denominator = np.sum(y_true) + np.sum(y_pred)
    if denominator == 0:
        return 0.0
    return numerator / denominator


def validate_mae(model, dataset, val_indices, criterion, device):
    """
    Validation 도시를 마스킹하여 모델 성능 평가.
    반환: (val_loss, rmse, cpc)
    """
    # Validation 마스크 구성
    val_mask_np = np.zeros(dataset.num_nodes, dtype=bool)
    val_mask_np[val_indices] = True

    X_static_masked = dataset.masking_static_features(dataset.X_static, val_indices, dataset.masking_indices)

    X_OD_masked = dataset.X_OD.copy()
    X_OD_masked[val_mask_np, :] = 0
    X_OD_masked[:, val_mask_np] = 0

    x_s = torch.tensor(X_static_masked, dtype=torch.float32, device=device).unsqueeze(0)
    x_d = torch.tensor(dataset.X_dist,  dtype=torch.float32, device=device).unsqueeze(0)
    x_o = torch.tensor(X_OD_masked, dtype=torch.float32, device=device).unsqueeze(0)
    y_o = torch.tensor(dataset.X_OD, dtype=torch.float32, device=device).unsqueeze(0)
    mask = torch.tensor(val_mask_np, dtype=torch.bool, device=device).unsqueeze(0)

    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.eval()

    with torch.no_grad():
        v_pred = model(x_s, x_o, x_d, mask)

    m2d = mask.unsqueeze(1) | mask.unsqueeze(2)
    v_loss = criterion(v_pred, y_o, 1.0, m2d).item()  # alpha=1.0, mask=m2d로 수정

    p_real = np.maximum(torch.expm1(v_pred[m2d]).cpu().numpy(), 0)
    y_real = torch.expm1(y_o[m2d]).cpu().numpy()

    rmse = np.sqrt(np.mean((y_real - p_real) ** 2))
    cpc = cpc_score(y_real, p_real)

    return v_loss, rmse, cpc