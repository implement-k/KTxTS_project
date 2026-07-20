import os
import sys
import numpy as np
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(current_dir)
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from dataset import ODDataset
from config import DONG_CODE_PATH, STATIC_DATA_PATH

def main():
    dataset = ODDataset(mode='test', use_log_transform=False)
    
    dong_df = pd.read_excel(DONG_CODE_PATH)
    dongs = dong_df['dong_code'].astype(int).values
    names = dong_df['dong_name'].values
    idx2name = {idx: name for idx, name in enumerate(names)}
    
    pred_path_v7 = os.path.join(current_dir, 'result', 'predicted_OD_matrix_mae_cpc:v7.csv')
    pred_matrix = pd.read_csv(pred_path_v7, header=None).values
    true_matrix = dataset.X_OD
    
    # 정적 변수 데이터프레임 로드
    static_df = pd.read_csv(STATIC_DATA_PATH)
    
    # 밀도 파생변수 생성
    static_df['worker_density'] = static_df['worker_count'] / (static_df['행정동전체면적_m2'] + 1e-5)
    static_df['business_density'] = static_df['business_count'] / (static_df['행정동전체면적_m2'] + 1e-5)
    static_df['station_density_지하철'] = static_df['station_count_지하철'] / (static_df['행정동전체면적_m2'] + 1e-5)
    
    test_indices = dataset.test_indices
    
    results = []
    
    for idx in test_indices:
        dong_name = idx2name.get(idx, str(idx))
        
        # Outgoing
        true_out = true_matrix[idx, :]
        pred_out = pred_matrix[idx, :]
        
        # 상위 타겟 정렬 (0~4: 최상위, 5~50: 중하위)
        sort_out_idx = np.argsort(true_out)[::-1]
        mid_targets = sort_out_idx[5:50]
        
        if len(mid_targets) > 0:
            log_true = np.log1p(true_out[mid_targets])
            log_pred = np.log1p(pred_out[mid_targets])
            male_5_50 = np.mean(np.abs(log_true - log_pred))
            mape_5_50 = np.mean(np.abs(true_out[mid_targets] - pred_out[mid_targets]) / (true_out[mid_targets] + 1e-5))
            mean_true = np.mean(true_out[mid_targets])
        else:
            male_5_50 = 0
            mape_5_50 = 0
            mean_true = 0
            
        results.append({
            'idx': idx,
            'dong_name': dong_name,
            'male_5_50': male_5_50,
            'mape_5_50': mape_5_50,
            'mean_true_5_50': mean_true
        })
        
    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values(by='mape_5_50', ascending=False)
    
    print("=== Top 10 Dongs with Highest Error (MAPE) in Targets 5~50 ===")
    print(res_df.head(10)[['dong_name', 'mape_5_50', 'male_5_50', 'mean_true_5_50']])
    
    print("\n=== Top 10 Dongs with Lowest Error (MAPE) in Targets 5~50 ===")
    print(res_df.tail(10)[['dong_name', 'mape_5_50', 'male_5_50', 'mean_true_5_50']])
    
    # Feature 비교
    high_err_indices = res_df.head(10)['idx'].values
    low_err_indices = res_df.tail(10)['idx'].values
    
    high_err_features = static_df.iloc[high_err_indices].mean(numeric_only=True)
    low_err_features = static_df.iloc[low_err_indices].mean(numeric_only=True)
    
    diff = (high_err_features - low_err_features) / (low_err_features.abs() + 1e-5)
    
    print("\n=== Feature Differences (High Error vs Low Error) ===")
    comp_df = pd.DataFrame({
        'High_Err_Mean': high_err_features,
        'Low_Err_Mean': low_err_features,
        'Diff_Ratio': diff
    })
    comp_df = comp_df.sort_values(by='Diff_Ratio', ascending=False)
    print(comp_df)

if __name__ == '__main__':
    main()
