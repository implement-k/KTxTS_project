import os
import sys
import numpy as np
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(current_dir)
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from dataset import ODDataset

def get_cpc(y_true, y_pred):
    mask = (y_true > 0) | (y_pred > 0)
    y_true_f = y_true[mask]
    y_pred_f = y_pred[mask]
    num = 2 * np.sum(np.minimum(y_true_f, y_pred_f))
    den = np.sum(y_true_f) + np.sum(y_pred_f)
    return num / den if den > 0 else 0.0

def main():
    dataset = ODDataset(mode='test', use_log_transform=False)
    
    pred_path_v7 = os.path.join(current_dir, 'result', 'predicted_OD_matrix_mae_cpc:v7.csv')
    pred_path_v2 = os.path.join(current_dir, 'result', 'predicted_OD_matrix_mae:v2.csv')
    
    pred_v7 = pd.read_csv(pred_path_v7, index_col=0, header=0).values
    pred_v2 = pd.read_csv(pred_path_v2, index_col=0, header=0).values
    
    true_matrix = dataset.X_OD
    
    # 평가 대상 인덱스 (test_indices에 해당하는 부분만)
    test_indices = dataset.test_indices
    true_test = true_matrix[test_indices, :]
    pred_v7_test = pred_v7[test_indices, :1137]
    pred_v2_test = pred_v2[test_indices, :1137]
    
    print("=== CPC Evaluation of Saved CSVs ===")
    print(f"v7 CSV CPC: {get_cpc(true_test, pred_v7_test):.4f}")
    print(f"v2 CSV CPC: {get_cpc(true_test, pred_v2_test):.4f}")
    
    print("\n--- Sanity Check for Row 0 (test_idx 0) ---")
    print(f"Row idx: {test_indices[0]}")
    
    # Sort true values to find large flows
    row_true = true_test[0]
    row_pred = pred_v7_test[0]
    top_idx = np.argsort(row_true)[::-1][:10]
    
    print("Top 10 True flows and their Predictions:")
    for idx in top_idx:
        print(f"  target {idx}: True={row_true[idx]:.2f}, Pred={row_pred[idx]:.2f}")

if __name__ == '__main__':
    main()
