import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import zscore

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OUTLIER_CONFIG

def detect_statistical_outliers(X_OD, dong_names, percentile=99.9):
    print(f"\n=== 1. 통계적 극단치 탐지 (상위 {100-percentile}%) ===")
    
    # Ignore diagonal (intrazonal) for general statistical check if needed, but here we include all
    threshold = np.percentile(X_OD, percentile)
    outliers = np.argwhere(X_OD > threshold)
    
    results = []
    for o_idx, d_idx in outliers:
        val = X_OD[o_idx, d_idx]
        results.append((dong_names[o_idx], dong_names[d_idx], val))
        
    results.sort(key=lambda x: x[2], reverse=True)
    
    print(f"기준값(Threshold): {threshold:.2f} 통행량 이상")
    for i, (o_name, d_name, val) in enumerate(results[:20]):
        print(f"  {i+1}. {o_name} -> {d_name} : {val:.1f} 통행량")
    return results

def detect_gravity_residuals(X_OD, X_dist, dong_names, top_percent=1.0):
    print(f"\n=== 2. 거리 대비 비정상적 통행량 (Gravity 잔차 상위 {top_percent}%) ===")
    
    # Simple Gravity Heuristic
    # O_pop roughly proportional to sum of outgoing
    O_tot = X_OD.sum(axis=1, keepdims=True)
    # D_pop roughly proportional to sum of incoming
    D_tot = X_OD.sum(axis=0, keepdims=True)
    
    # Avoid 0
    O_tot = np.clip(O_tot, 1, None)
    D_tot = np.clip(D_tot, 1, None)
    dist = np.clip(X_dist, 1, None)
    
    # Expected OD proportional to O_tot * D_tot / dist^2
    expected = (O_tot * D_tot) / (dist ** 2)
    
    # Scale expected to match sum of X_OD
    scale = X_OD.sum() / expected.sum()
    expected = expected * scale
    
    # Calculate Residuals (Absolute Difference)
    residuals = np.abs(X_OD - expected)
    
    threshold = np.percentile(residuals, 100 - top_percent)
    outliers = np.argwhere(residuals > threshold)
    
    results = []
    for o_idx, d_idx in outliers:
        res = residuals[o_idx, d_idx]
        actual = X_OD[o_idx, d_idx]
        exp = expected[o_idx, d_idx]
        results.append((dong_names[o_idx], dong_names[d_idx], actual, exp, res))
        
    results.sort(key=lambda x: x[4], reverse=True)
    
    for i, (o_name, d_name, actual, exp, res) in enumerate(results[:20]):
        print(f"  {i+1}. {o_name} -> {d_name} : 실제 {actual:.1f} / 예상 {exp:.1f} (오차 {res:.1f})")
    return results

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, 'dataset')
    
    print("Loading data...")
    X_OD = np.load(os.path.join(data_dir, 'X_OD_2D.npy'))
    X_dist = np.load(os.path.join(data_dir, 'X_distance.npy'))
    
    dong_file = os.path.join(base_dir, '..', 'dataset', 'dong_code.xlsx')
    dong_df = pd.read_excel(dong_file)
    dong_names = (dong_df['시군구'].astype(str) + ' ' + dong_df['name'].astype(str)).values
    
    detect_statistical_outliers(X_OD, dong_names, OUTLIER_CONFIG['statistical_percentile'])
    detect_gravity_residuals(X_OD, X_dist, dong_names, OUTLIER_CONFIG['residual_top_percent'])

if __name__ == '__main__':
    main()
