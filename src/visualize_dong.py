import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 설정 (Mac)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

current_dir = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(current_dir)
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from dataset import ODDataset
from config import DONG_CODE_PATH

def main():
    # 데이터셋 로드 (원래 스케일의 OD 매트릭스 얻기 위함)
    dataset = ODDataset(mode='test', use_log_transform=False)
    
    # 동 이름 매핑 가져오기
    
    
    dong_df = pd.read_excel(DONG_CODE_PATH)
    dongs = dong_df['dong_code'].astype(int).values
    names = dong_df['dong_name'].values
    idx2code = {idx: code for idx, code in enumerate(dongs)}
    code2name = dict(zip(dongs, names))
    
    pred_path_v7 = os.path.join(current_dir, 'result', 'predicted_OD_matrix_mae_cpc:v7.csv')
    
    if not os.path.exists(pred_path_v7):
        print("Error: Required prediction CSV not found.")
        return
        
    pred_matrix_v7 = pd.read_csv(pred_path_v7, index_col=0, header=0).values
    true_matrix = dataset.X_OD
    
    test_indices = dataset.test_indices
    
    # 테스트 행정동 중 통행량(O+D)이 가장 많은 상위 4개 선택
    test_flows = []
    for idx in test_indices:
        total_flow = true_matrix[idx, :].sum() + true_matrix[:, idx].sum()
        test_flows.append((idx, total_flow))
    test_flows.sort(key=lambda x: x[1], reverse=True)
    top_4_indices = [x[0] for x in test_flows[:4]]
    
    fig, axes = plt.subplots(4, 2, figsize=(15, 20))
    
    for i, idx in enumerate(top_4_indices):
        dong_code = idx2code.get(idx, str(idx))
        dong_name = code2name.get(dong_code, str(dong_code))
        
        # 1. Outgoing (출발) 통행량 비교
        true_out = true_matrix[idx, :]
        pred_out_v7 = pred_matrix_v7[idx, :]
        
        # 통행량이 많은 상위 50개 타겟 동만 정렬해서 시각화
        sort_out_idx = np.argsort(true_out)[::-1][:50]
        
        axes[i, 0].plot(true_out[sort_out_idx], label='True (실제)', color='blue', alpha=0.7, marker='o', markersize=3)
        axes[i, 0].plot(pred_out_v7[sort_out_idx], label='mae:v7 (하이브리드) 예측', color='red', alpha=0.7, marker='x', markersize=3)
        axes[i, 0].set_title(f'[{dong_name}] 출발(Outgoing) 통행량 - 상위 50개 타겟')
        axes[i, 0].set_ylabel('통행량')
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)
        
        # 2. Incoming (도착) 통행량 비교
        true_in = true_matrix[:, idx]
        pred_in_v7 = pred_matrix_v7[:, idx]
        
        sort_in_idx = np.argsort(true_in)[::-1][:50]
        
        axes[i, 1].plot(true_in[sort_in_idx], label='True (실제)', color='blue', alpha=0.7, marker='o', markersize=3)
        axes[i, 1].plot(pred_in_v7[sort_in_idx], label='mae:v7 (하이브리드) 예측', color='red', alpha=0.7, marker='x', markersize=3)
        axes[i, 1].set_title(f'[{dong_name}] 도착(Incoming) 통행량 - 상위 50개 타겟')
        axes[i, 1].set_ylabel('통행량')
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3)
        
    plt.tight_layout()
    save_path = os.path.join(current_dir, 'result', 'test_dong_visualization.png')
    plt.savefig(save_path, dpi=150)
    print(f"Visualization saved to {save_path}")

if __name__ == '__main__':
    main()
