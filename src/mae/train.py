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
import random

from mae.dataset import ODDataset
from mae.models import SpatialODMAE
from tqdm import tqdm
from mae.loss import WeightedMSELossWrapper

def create_k_folds(train_indices, X_dist, k_fold=3):
    unassigned = set(train_indices)
    val_cities = []

    while len(unassigned) > 0:
        if len(unassigned) <= 8:
            val_cities.append(list(unassigned))
            break
        
        val_city_center = random.choice(list(unassigned))
        val_city_size = random.randint(4, 8)

        # validation city center와 unassigned nodes 간의 거리 계산
        unassigned_list  = list(unassigned)
        dists = X_dist[val_city_center, unassigned_list]
        
        # validation city size만큼 가장 가까운 노드 선택
        nearest_idx = np.argsort(dists)[:val_city_size]
        val_city = [unassigned_list[i] for i in nearest_idx]

        # validation city 추가 및 unassigned에서 제거
        val_cities.append(val_city)
        for dong in val_city:
            unassigned.remove(dong)

    # shuffle하고 fold로 나누기
    random.shuffle(val_cities)
    folds = [[] for _ in range(k_fold)]
    for i, val_city in enumerate(val_cities):
        folds[i % k_fold].extend(val_city)

    return [np.array(f, dtype=int) for f in folds]


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
    v_loss = criterion(v_pred, y_o, m2d).item()

    p_real = np.maximum(torch.expm1(v_pred[m2d]).cpu().numpy(), 0)
    y_real = torch.expm1(y_o[m2d]).cpu().numpy()

    rmse = np.sqrt(np.mean((y_real - p_real) ** 2))
    cpc = cpc_score(y_real, p_real)

    return v_loss, rmse, cpc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=TRAIN_CONFIG['epochs'])
    parser.add_argument('--batch_size', type=int, default=TRAIN_CONFIG['batch_size'])
    parser.add_argument('--k_fold', type=int, default=3, help='K-Fold 수')
    args = parser.parse_args()

    dataset = ODDataset(mode='train')
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps'  if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # K-Fold 생성 (test 도시 제외한 train 도시들 대상)
    folds = create_k_folds(dataset.train_indices, dataset.X_dist, k_fold=args.k_fold)

    min_mask = TRAIN_CONFIG['min_mask_size']
    max_mask = TRAIN_CONFIG['max_mask_size']

    cv_rmses = []
    cv_cpcs  = []

    for fold in range(args.k_fold):
        print(f"\n{'='*20} FOLD {fold+1}/{args.k_fold} {'='*20}")
        val_indices = folds[fold]

        # 이번 fold에서의 train_indices (test + val 제외)
        fold_train_mask = np.ones(dataset.num_nodes, dtype=bool)
        fold_train_mask[dataset.test_indices] = False
        fold_train_mask[val_indices] = False
        dataset.train_indices = np.where(fold_train_mask)[0]

        train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        model = SpatialODMAE(num_nodes=dataset.num_nodes,num_features=dataset.X_static.shape[1]).to(device)
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
        best_model_path = f'best_model_mae_fold_{fold+1}.pth'

        for epoch in range(args.epochs):
            progress = epoch / max(1, args.epochs - 1)
            current_mask_size = int(min_mask + (max_mask - min_mask) * progress)
            dataset.max_mask_size = current_mask_size
            current_alpha = max(1.0, 10.0 * (1.0 - progress))

            model.train()
            train_loss = 0

            pbar = tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch+1}/{args.epochs} " f"[Mask:{current_mask_size} α:{current_alpha:.1f}]")
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

            print(f"Fold {fold+1} Epoch {epoch+1} Train Loss: {train_loss/len(train_loader):.4f}")

            # Validation (2 epoch 마다)
            if epoch % 2 == 1 or epoch == args.epochs - 1:
                v_loss, rmse, cpc = validate(model, dataset, val_indices, criterion, device)
                print(f"  ➜ [Val] Loss: {v_loss:.4f} | RMSE: {rmse:.2f} | CPC: {cpc:.4f}")

                if rmse < best_val_rmse:
                    best_val_rmse = rmse
                    best_cpc      = cpc
                    current_dir   = os.path.dirname(os.path.abspath(__file__))
                    torch.save(model.state_dict(),
                               os.path.join(current_dir, best_model_path))
                    print(f"  ➜ [Checkpoint] Best saved! (RMSE:{rmse:.2f} CPC:{cpc:.4f})")

                model.train()

        cv_rmses.append(best_val_rmse)
        cv_cpcs.append(best_cpc)
        print(f"Fold {fold+1} 완료  Best RMSE: {best_val_rmse:.2f} | CPC: {best_cpc:.4f}")

    print(f"\n{'='*40}")
    print(f"K-FOLD CV RESULTS")
    for i, (r, c) in enumerate(zip(cv_rmses, cv_cpcs)):
        print(f"  Fold {i+1}: RMSE={r:.2f}  CPC={c:.4f}")
    print(f"  평균  : RMSE={np.mean(cv_rmses):.2f}  CPC={np.mean(cv_cpcs):.4f}")
    print(f"{'='*40}")


if __name__ == '__main__':
    main()
