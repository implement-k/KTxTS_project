import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRAIN_CONFIG

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import joblib
import matplotlib.pyplot as plt

from dataset import ODDataset
from models import DeepGravity, SpatialODMAE1, SpatialODMAE5
from loss import LOSS_REGISTRY
from eval_utils import evaluate_breakdown, get_git_commit_hash

def cpc_score(y_true, y_pred):
    numerator = 2 * np.sum(np.minimum(y_true, y_pred))
    denominator = np.sum(y_true) + np.sum(y_pred)
    if denominator == 0: return 0.0
    return numerator / denominator

def visualize_predictions(y_true, y_pred, model_name):
    plt.figure(figsize=(12, 5))
    
    # Scatter Plot
    plt.subplot(1, 2, 1)
    # 0값 처리를 위해 log1p 사용
    plt.scatter(np.log1p(y_true), np.log1p(y_pred), alpha=0.3, s=2)
    plt.plot([0, 10], [0, 10], 'r--')
    plt.xlabel('True OD (log1p)')
    plt.ylabel('Predicted OD (log1p)')
    plt.title(f'Scatter Plot ({model_name})')
    
    # Residual Plot
    plt.subplot(1, 2, 2)
    residual = y_pred - y_true
    plt.scatter(np.log1p(y_true), residual, alpha=0.3, s=2)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel('True OD (log1p)')
    plt.ylabel('Residual (Pred - True)')
    plt.title('Residual Plot (Heavy-tail Check)')
    
    plt.tight_layout()
    save_path = f'results_{model_name}.png'
    plt.savefig(save_path)
    plt.close()
    print(f"Visualization saved to {save_path}")

# 공식 비교 지표는 RMSE/MAE/CPC(real-scale)이며, 여기서는 loss 값 자체를 계산하지 않는다.
# 향후 args.loss에 맞는 test loss(log-scale)가 별도로 필요해지면 아래처럼 LOSS_REGISTRY로 구성할 수 있다.
#   from loss import LOSS_REGISTRY
#   test_criterion = LOSS_REGISTRY[args.loss](...).to(device)
#   test_loss = test_criterion(pred[mask_2d], y_od[mask_2d], None).item()

def test_dl_model(args, test_dataset):
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if args.model == 'mae5':
        model = SpatialODMAE5(num_nodes=test_dataset.num_nodes, num_features=test_dataset.X_static.shape[1]).to(device)
    elif args.model == 'mae1':
        model = SpatialODMAE1(num_nodes=test_dataset.num_nodes, num_features=test_dataset.X_static.shape[1]).to(device)
    else:
        model = DeepGravity(num_features=test_dataset.X_static.shape[1]).to(device)
        
    best_model_path = f'best_model_{args.model}_{args.loss}_seed{args.seed}.pth'

    if not os.path.exists(best_model_path):
        print(f"Error: {best_model_path} not found! Please train the model first.")
        return
        
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    print(f"Loaded {best_model_path} for testing.")
        
    model.train()
    for m_module in model.modules():
        if isinstance(m_module, torch.nn.Dropout):
            m_module.eval()
            
    all_y_true = []
    all_y_pred = []
    all_O_idx = []
    all_D_idx = []
    all_diag_y_true = []
    all_diag_y_pred = []

    # O/D 행정동 인덱스 그리드 (동일동/타동 구분 및 diagnostic용, 배치마다 동일하므로 루프 밖에서 1회 계산)
    O_idx_grid, D_idx_grid = np.indices((test_dataset.num_nodes, test_dataset.num_nodes))

    with torch.no_grad():
        for batch in test_loader:
            x_static = batch['X_static'].to(device)
            x_dist = batch['X_dist'].to(device)
            mask = batch['mask'].to(device)

            x_od_masked = batch['X_OD_masked'].to(device)
            y_od = batch['y_OD'].to(device)

            if args.model in ['mae1', 'mae5']:
                pred = model(x_static, x_od_masked, x_dist, mask)
            else:
                pred = model(x_static, x_dist)

            mask_2d = mask.unsqueeze(1) | mask.unsqueeze(2)

            if args.model == 'mae5':
                m_exp = mask_2d.unsqueeze(-1).expand_as(y_od)
                pred_real = torch.expm1(pred[m_exp]).cpu().numpy()
                y_real = torch.expm1(y_od[m_exp]).cpu().numpy()
                # mae5는 (N,N,5) 채널 구조라 이번 breakdown(2D O/D 기준) 범위 밖 — 전체지표만 계산
            else:
                pred_real = torch.expm1(pred[mask_2d]).cpu().numpy()
                y_real = torch.expm1(y_od[mask_2d]).cpu().numpy()

                # 공식 평가(테스트 지역 마스킹 대상)에 대응하는 O/D 인덱스를 같은 순서로 누적
                # (배치가 여러 개여도 all_y_true/all_y_pred와 동일하게 append 후 마지막에 concatenate)
                mask_2d_np = mask_2d[0].cpu().numpy()
                all_O_idx.append(O_idx_grid[mask_2d_np])
                all_D_idx.append(D_idx_grid[mask_2d_np])

                # 참고용 diagnostic_full_matrix: 전체 N×N (마스킹 여부와 무관, 마스킹되지 않은 입력 정보가
                # 노출된 상태이므로 공식 지표와 절대 섞지 않음). 배치별로 누적 후 마지막에 concatenate.
                # (모델은 mask 여부와 무관하게 항상 전체 N×N을 예측하므로 forward pass 추가 불필요)
                all_diag_y_pred.append(np.maximum(torch.expm1(pred[0]).cpu().numpy(), 0).flatten())
                all_diag_y_true.append(torch.expm1(y_od[0]).cpu().numpy().flatten())

            pred_real = np.maximum(pred_real, 0)

            all_y_true.append(y_real)
            all_y_pred.append(pred_real)

    all_y_true = np.concatenate(all_y_true)
    all_y_pred = np.concatenate(all_y_pred)

    rmse = np.sqrt(np.mean((all_y_true - all_y_pred)**2))
    cpc = cpc_score(all_y_true, all_y_pred)
    mae = float(np.mean(np.abs(all_y_true - all_y_pred)))

    print(f"\n=== Test Results ({args.model}) ===")
    print(f"RMSE (Real scale, 테스트 지역 마스킹 대상): {rmse:.2f}")
    print(f"MAE (Real scale, 테스트 지역 마스킹 대상): {mae:.2f}")
    print(f"CPC (Common Part of Commuters, 테스트 지역 마스킹 대상): {cpc:.4f}")

    visualize_predictions(all_y_true, all_y_pred, args.model)

    if all_O_idx:
        all_O_idx = np.concatenate(all_O_idx)
        all_D_idx = np.concatenate(all_D_idx)
        diagnostic_full = (np.concatenate(all_diag_y_true), np.concatenate(all_diag_y_pred))

        # loss/파이프라인별 실제 alpha 설정을 정확히 기록 (train.py의 실제 값과 반드시 일치시킬 것)
        ALPHA_INFO = {
            'weighted_mse': 'alpha=1.5 fixed (WeightedMSELossWrapper 기본값, src/model/train.py와 동일)',
        }
        run_meta = {
            'pipeline': args.model,
            'loss': args.loss,
            'seed': args.seed,
            'epochs': getattr(args, 'epochs', None),
            'batch_size': getattr(args, 'batch_size', None),
            'alpha_info': ALPHA_INFO.get(args.loss, 'unknown'),
            'git_commit': get_git_commit_hash(),
            'device': str(device),
        }
        csv_path = f'results_{args.model}_{args.loss}_seed{args.seed}.csv'
        evaluate_breakdown(all_y_true, all_y_pred, all_O_idx, all_D_idx, run_meta, csv_path,
                            diagnostic_full=diagnostic_full)


def test_tabular_model(args, test_dataset):
    X_OD_real = test_dataset.X_OD
    X_dist_real = test_dataset.X_dist
    X_static = test_dataset.X_static
    num_nodes = test_dataset.num_nodes
    
    O_idx, D_idx = np.indices((num_nodes, num_nodes))
    O_idx = O_idx.flatten()
    D_idx = D_idx.flatten()
    
    y = X_OD_real.flatten()
    dist = X_dist_real.flatten()
    
    O_pop_total = X_OD_real.sum(axis=1)
    D_pop_total = X_OD_real.sum(axis=0)
    O_pop = O_pop_total[O_idx]
    D_pop = D_pop_total[D_idx]
    
    save_path = f'best_model_{args.model}.pkl'
    if not os.path.exists(save_path):
        print(f"Error: {save_path} not found! Please train the model first.")
        return
        
    print(f"Loading {save_path}...")
    model = joblib.load(save_path)
    print("Predicting...")
            
    if args.model == 'lgbm':
        O_stat_feat = X_static[O_idx]
        D_stat_feat = X_static[D_idx]
        X_tabular = np.column_stack([dist, O_stat_feat, D_stat_feat])
        
        test_cities = set(test_dataset.test_indices)
        is_test = np.array([(o in test_cities) or (d in test_cities) for o, d in zip(O_idx, D_idx)])
        
        y_pred = model.predict(X_tabular[is_test])
        y_pred = np.maximum(y_pred, 0)
        y_true = y[is_test]
    
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    cpc = cpc_score(y_true, y_pred)
    
    print(f"\n=== Test Results ({args.model}) ===")
    print(f"Test size: {len(y_true)} OD pairs")
    print(f"RMSE (Real scale): {rmse:.2f}")
    print(f"CPC (Common Part of Commuters): {cpc:.4f}")
    
    visualize_predictions(y_true, y_pred, args.model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=TRAIN_CONFIG['model_type'], choices=['lgbm', 'deep_gravity', 'mae1', 'mae5'])
    parser.add_argument('--loss', type=str, default='weighted_mse', choices=list(LOSS_REGISTRY.keys()))
    parser.add_argument('--seed', type=int, default=42)
    # 아래 둘은 체크포인트 탐색에는 쓰이지 않고, 단독 실행 시 결과 CSV 메타데이터 기록용
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    args = parser.parse_args()
    
    channel = 5 if args.model == 'mae5' else 1
    test_dataset = ODDataset(mode='test', channel=channel, isLogScale=True if args.model in ['mae1', 'mae5', 'deep_gravity'] else False)
    
    if args.model in ['mae1', 'mae5', 'deep_gravity']:
        test_dl_model(args, test_dataset)
    elif args.model in ['lgbm']:
        test_tabular_model(args, test_dataset)

if __name__ == '__main__':
    main()