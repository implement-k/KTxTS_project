import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRAIN_CONFIG

import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from mae.dataset import ODDataset
from mae.models import SpatialODMAE
from tqdm import tqdm
from mae.loss import WeightedMSELossWrapper

def cpc_score(y_true, y_pred):
    numerator   = 2 * np.sum(np.minimum(y_true, y_pred))
    denominator = np.sum(y_true) + np.sum(y_pred)
    if denominator == 0:
        return 0.0
    return numerator / denominator


def validate(model, dataset, val_indices, criterion, device):
    """
    Validation 도시를 마스킹하여 모델 성능 평가.
    반환: (val_loss, rmse, cpc)
    """
    # Validation 마스크 구성
    val_mask_np = np.zeros(dataset.num_nodes, dtype=bool)
    val_mask_np[val_indices] = True

    X_static_masked = dataset.X_static.copy()
    for m_idx in dataset.masking_indices:
        X_static_masked[val_indices, m_idx] = 0.0
    X_static_masked[val_indices, -1] = 1.0  

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=TRAIN_CONFIG['epochs'])
    parser.add_argument('--batch_size', type=int, default=TRAIN_CONFIG['batch_size'])
    args = parser.parse_args()

    dataset = ODDataset(mode='train')
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps'  if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    min_mask = TRAIN_CONFIG['min_mask_size']
    max_mask = TRAIN_CONFIG['max_mask_size']

    print(f"\n{'='*20} Fast Validation Mode (Test Set) {'='*20}")
    
    # Validation 대상은 Test 도시 전체
    val_indices = dataset.test_indices

    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = SpatialODMAE(num_nodes=dataset.num_nodes, num_features=dataset.X_static.shape[1]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    total_steps = args.epochs * len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=5e-4, 
        total_steps=total_steps,
        pct_start=0.3, 
        anneal_strategy='cos'
    )
    criterion = WeightedMSELossWrapper().to(device)

    best_val_rmse = float('inf')
    best_cpc = 0.0
    best_model_path = 'best_model_mae.pth'

    for epoch in range(args.epochs):
        progress = epoch / max(1, args.epochs - 1)
        current_mask_size = int(min_mask + (max_mask - min_mask) * progress)
        dataset.max_mask_size = current_mask_size
        current_alpha = max(1.0, 10.0 * (1.0 - progress))

        model.train()
        train_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} " f"[Mask:{current_mask_size} α:{current_alpha:.1f}]")
        for batch in pbar:
            x_static = batch['X_static'].to(device)
            x_dist = batch['X_dist'].to(device)
            mask = batch['mask'].to(device)
            x_od_masked = batch['X_OD_masked'].to(device)
            y_od = batch['y_OD'].to(device)

            optimizer.zero_grad()
            pred = model(x_static, x_od_masked, x_dist, mask)
            mask_2d = mask.unsqueeze(1) | mask.unsqueeze(2)
            loss = criterion(pred, y_od, current_alpha, mask_2d)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{scheduler.get_last_lr()[0]:.1e}"})

        print(f"Epoch {epoch+1} Train Loss: {train_loss/len(train_loader):.4f}")

        # Validation (2 epoch 마다)
        if epoch % 2 == 1 or epoch == args.epochs - 1:
            v_loss, rmse, cpc = validate(model, dataset, val_indices, criterion, device)
            print(f"  ➜ [Val] Loss: {v_loss:.4f} | RMSE: {rmse:.2f} | CPC: {cpc:.4f}")

            if rmse < best_val_rmse:
                best_val_rmse = rmse
                best_cpc = cpc
                current_dir = os.path.dirname(os.path.abspath(__file__))
                torch.save(model.state_dict(),
                           os.path.join(current_dir, best_model_path))
                print(f"  ➜ [Checkpoint] Best saved! (RMSE:{rmse:.2f} CPC:{cpc:.4f})")

            model.train()

    print(f"\nTraining Complete. Best RMSE: {best_val_rmse:.2f} | CPC: {best_cpc:.4f}")

if __name__ == '__main__':
    main()
