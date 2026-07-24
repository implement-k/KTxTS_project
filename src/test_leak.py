import torch
from dataset import ODDataset
from mae.models import SpatialODMAE
from validation import validate_mae
from loss import WeightedMSELoss

def main():
    device = torch.device('cpu')
    dataset = ODDataset(mode='train')
    model = SpatialODMAE(num_features=dataset.X_static.shape[1]).to(device)
    criterion = WeightedMSELoss().to(device)
    
    v_loss, rmse, cpc = validate_mae(model, dataset, dataset.test_indices, criterion, device)
    print(f"Initial Validation -> Loss: {v_loss:.4f} | RMSE: {rmse:.2f} | CPC: {cpc:.4f}")

if __name__ == '__main__':
    main()
