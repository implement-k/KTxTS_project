import os
import torch
import numpy as np
from torch.utils.data import DataLoader

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from dataset import ODDataset
from mae.models import SpatialODMAE

def estimate_loss_contribution():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 학습용 데이터(train mode)에서 1개 배치만 뽑아서 확인
    test_dataset = ODDataset(mode='train')
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    diag_loss_sum = 0.0
    offdiag_loss_sum = 0.0
    N = test_dataset.num_nodes
    batch = next(iter(test_loader))
    with torch.no_grad():
        x_static = batch['X_static'].to(device)
        mask = batch['mask'].to(device)
        y_od = batch['y_OD'].to(device)

        B = y_od.shape[0]

        # 대각 성분 마스크 생성
        diag_mask = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0).expand(B, -1, -1)
        mask_2d = mask.unsqueeze(1) | mask.unsqueeze(2)

        valid_diag_mask = diag_mask & mask_2d
        valid_offdiag_mask = (~diag_mask) & mask_2d
        
        # 순수 타겟(y_od)의 제곱합(Scale) 비교
        y_diag = y_od[valid_diag_mask]
        y_offdiag = y_od[valid_offdiag_mask]
        
        d_loss = torch.nansum(y_diag**2).item()
        od_loss = torch.nansum(y_offdiag**2).item()

        diag_loss_sum += d_loss
        offdiag_loss_sum += od_loss
        
        d_count = (~torch.isnan(y_diag)).sum().item()
        od_count = (~torch.isnan(y_offdiag)).sum().item()

    print(f"\n[Target (y_OD) Squared Sum in 1 Batch (Size: {B})]")
    print(f"diagonal 타겟(y_od^2) 총합: {diag_loss_sum:.2e} (유효 원소 개수: {d_count:,})")
    print(f"off-diagonal 타겟(y_od^2) 총합: {offdiag_loss_sum:.2e} (유효 원소 개수: {od_count:,})")
    if offdiag_loss_sum > 0:
        ratio = diag_loss_sum / offdiag_loss_sum
        print(f"총합 비율 (Diag / Off-diag): {ratio:.4f}")
        print(f"단일 원소당 평균 크기 비교: Diag({diag_loss_sum/d_count:.4f}) vs Off-diag({offdiag_loss_sum/od_count:.4f})")

estimate_loss_contribution()
