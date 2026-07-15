import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import joblib
import matplotlib.pyplot as plt
from model import Stage2Model
from dataset import ODDataset
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
#   test_loss = test_criterion(pred[mask_2d], y_od_log[mask_2d], None).item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss', type=str, default='weighted_mse', choices=list(LOSS_REGISTRY.keys()))
    parser.add_argument('--seed', type=int, default=42)
    # 체크포인트 탐색에는 쓰이지 않고, 단독 실행 시 결과 CSV 메타데이터 기록용
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    args = parser.parse_args()

    test_dataset = ODDataset(mode='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = Stage2Model(num_features=test_dataset.X_static.shape[1]).to(device)
    best_model_path = f'best_model_twostage_{args.loss}_seed{args.seed}.pth'

    if not os.path.exists(best_model_path):
        print(f"Error: {best_model_path} not found! Please train the model first.")
        return

    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    print(f"Loaded {best_model_path} for testing.")

    model.train()
    for m_module in model.modules():
        if isinstance(m_module, torch.nn.Dropout):
            m_module.eval()

    model_self = joblib.load(f'lgbm_self_seed{args.seed}.pkl')
    model_inter = joblib.load(f'lgbm_inter_seed{args.seed}.pkl')
    log_self_all = model_self.predict(test_dataset.X_static)
    log_inter_all = model_inter.predict(test_dataset.X_static)
    log_self_tensor = torch.tensor(log_self_all, dtype=torch.float32, device=device).unsqueeze(0)
    log_inter_tensor = torch.tensor(log_inter_all, dtype=torch.float32, device=device).unsqueeze(0)

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

            y_od = batch['y_OD'].to(device)

            pred = model(x_static, x_dist, log_self_tensor, log_inter_tensor)

            mask_2d = mask.unsqueeze(1) | mask.unsqueeze(2)

            pred_real = torch.expm1(pred[mask_2d]).cpu().numpy()
            y_real = y_od[mask_2d].cpu().numpy()

            pred_real = np.maximum(pred_real, 0)

            all_y_true.append(y_real)
            all_y_pred.append(pred_real)

            # 공식 평가(테스트 지역 마스킹 대상)에 대응하는 O/D 인덱스를 같은 순서로 누적
            # (배치가 여러 개여도 all_y_true/all_y_pred와 동일하게 append 후 마지막에 concatenate)
            mask_2d_np = mask_2d[0].cpu().numpy()
            all_O_idx.append(O_idx_grid[mask_2d_np])
            all_D_idx.append(D_idx_grid[mask_2d_np])

            # 참고용 diagnostic_full_matrix: 전체 N×N (마스킹 여부와 무관, 마스킹되지 않은 입력 정보가
            # 노출된 상태이므로 공식 지표와 절대 섞지 않음). 배치별로 누적 후 마지막에 concatenate.
            # (모델은 mask 여부와 무관하게 항상 전체 N×N을 예측하므로 forward pass 추가 불필요)
            all_diag_y_pred.append(np.maximum(torch.expm1(pred[0]).cpu().numpy(), 0).flatten())
            all_diag_y_true.append(y_od[0].cpu().numpy().flatten())

    all_y_true = np.concatenate(all_y_true)
    all_y_pred = np.concatenate(all_y_pred)
    all_O_idx = np.concatenate(all_O_idx)
    all_D_idx = np.concatenate(all_D_idx)
    diagnostic_full = (np.concatenate(all_diag_y_true), np.concatenate(all_diag_y_pred))

    rmse = np.sqrt(np.mean((all_y_true - all_y_pred)**2))
    cpc = cpc_score(all_y_true, all_y_pred)
    mae = float(np.mean(np.abs(all_y_true - all_y_pred)))

    print(f"\n=== Test Results (twostage) ===")
    print(f"RMSE (Real scale, 테스트 지역 마스킹 대상): {rmse:.2f}")
    print(f"MAE (Real scale, 테스트 지역 마스킹 대상): {mae:.2f}")
    print(f"CPC (Common Part of Commuters, 테스트 지역 마스킹 대상): {cpc:.4f}")

    visualize_predictions(all_y_true, all_y_pred, "twostage")

    # loss/파이프라인별 실제 alpha 설정을 정확히 기록 (train.py의 실제 값과 반드시 일치시킬 것)
    ALPHA_INFO = {
        'weighted_mse': 'alpha schedule 10.0->1.0 (linear over epochs, src/twostage/train.py의 기존 동작 유지)',
    }
    run_meta = {
        'pipeline': 'twostage',
        'loss': args.loss,
        'seed': args.seed,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'alpha_info': ALPHA_INFO.get(args.loss, 'unknown'),
        'git_commit': get_git_commit_hash(),
        'device': str(device),
    }
    csv_path = f'results_twostage_{args.loss}_seed{args.seed}.csv'
    evaluate_breakdown(all_y_true, all_y_pred, all_O_idx, all_D_idx, run_meta, csv_path,
                        diagnostic_full=diagnostic_full)

if __name__ == '__main__':
    main()