import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRAIN_CONFIG

import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import ODDataset
from mae.models import SpatialODMAE
from tqdm import tqdm
from loss import WeightedMSELoss, HybridWeightedMSELoss, HuberLoss
import wandb
from validation import validate_mae

def str2bool(v):
    return str(v).lower() in ("yes", "true", "t", "1")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=TRAIN_CONFIG['epochs'])
    parser.add_argument('--batch_size', type=int, default=TRAIN_CONFIG['batch_size'])
    parser.add_argument('--loss_type', type=str, default='weighted_mse', choices=['weighted_mse', 'hybrid', 'huber']) # v1, v2, v3: weighted_mse
    parser.add_argument('--od_embed_layers', type=int, default=3)                   # v1: 1, v2: 2, v3, v4: 3
    parser.add_argument('--use_friction', type=str2bool, default=True)              # v1, v2, v3: False, v4: True
    parser.add_argument('--use_self_loop_predictor', type=str2bool, default=True)   # v1: False, v2, v3, v4: True
    parser.add_argument('--use_wandb', type=str2bool, default=False)
    args = parser.parse_args()
    
    if args.use_wandb: wandb.init(project="SpatialODMAE", config=vars(args))
    
    print("선택된 argument:")
    for arg in vars(args): print(f"  {arg}: {getattr(args, arg)}")

    dataset = ODDataset(mode='train')
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps'  if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    min_mask = TRAIN_CONFIG['min_mask_size']
    max_mask = TRAIN_CONFIG['max_mask_size']

    # Validation 대상은 Test 도시 전체
    val_indices = dataset.test_indices
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = SpatialODMAE(num_nodes=dataset.num_nodes, 
                         num_features=dataset.X_static.shape[1],
                         od_embed_layers=args.od_embed_layers,
                         use_distance_friction=args.use_friction,
                         use_self_loop_predictor=args.use_self_loop_predictor).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    total_steps = args.epochs * len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=5e-4, 
        total_steps=total_steps,
        pct_start=0.3, 
        anneal_strategy='cos'
    )
    
    if args.loss_type == 'hybrid': criterion = HybridWeightedMSELoss().to(device)
    elif args.loss_type == 'huber': criterion = HuberLoss().to(device)
    else: criterion = WeightedMSELoss().to(device)

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

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")

        # Validation (2 epoch 마다)
        if epoch % 2 == 1 or epoch == args.epochs - 1:
            v_loss, rmse, cpc = validate_mae(model, dataset, val_indices, criterion, device)
            print(f"  ➜ [Val] Loss: {v_loss:.4f} | RMSE: {rmse:.2f} | CPC: {cpc:.4f}")
            
            if args.use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "val_loss": v_loss,
                    "val_rmse": rmse,
                    "val_cpc": cpc
                })

            if rmse < best_val_rmse:
                best_val_rmse = rmse
                current_dir = os.path.dirname(os.path.abspath(__file__))
                torch.save(model.state_dict(),
                           os.path.join(current_dir, best_model_path))
                print(f"  ➜ [Checkpoint] Best RMSE saved! (RMSE:{rmse:.2f} CPC:{cpc:.4f})")

            if cpc > best_cpc:
                best_cpc = cpc
                current_dir = os.path.dirname(os.path.abspath(__file__))
                torch.save(model.state_dict(),
                           os.path.join(current_dir, 'best_model_mae_cpc.pth'))
                print(f"  ➜ [Checkpoint] Best CPC saved! (RMSE:{rmse:.2f} CPC:{cpc:.4f})")

            model.train()

    print(f"\nTraining Complete. Best RMSE: {best_val_rmse:.2f} | Best CPC: {best_cpc:.4f}")
    
    if args.use_wandb: wandb.finish()

if __name__ == '__main__':
    main()
