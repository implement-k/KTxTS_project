import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# config.py 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OD_DATA_PATH, DONG_CODE_PATH, DIST_DATA_PATH

def main():
    print("데이터 로딩 중...")
    
    # 1. 행정동 코드 로드 및 인덱스 매핑
    dong_df = pd.read_excel(DONG_CODE_PATH)
    dongs = dong_df['dong_code'].astype(int).values
    num_nodes = len(dongs)
    dong2idx = {code: i for i, code in enumerate(dongs)}

    # 2. OD 데이터 로드
    od_df = pd.read_csv(OD_DATA_PATH)
    o_indices = od_df['O_dong_code'].map(dong2idx).values
    d_indices = od_df['D_dong_code'].map(dong2idx).values
    valid_mask = pd.notna(o_indices) & pd.notna(d_indices)
    
    o_idx_valid = o_indices[valid_mask].astype(int)
    d_idx_valid = d_indices[valid_mask].astype(int)
    
    purposes = ['귀가', '출근', '등교', '업무', '기타']
    calculated_total = od_df[purposes].sum(axis=1)
    
    # 전체 N x N OD 매트릭스 구성
    X_OD = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    X_OD[o_idx_valid, d_idx_valid] = calculated_total.values[valid_mask]
    
    # =====================================================================
    # 1. OD 0이 아닌 유효 관측치 비율 및 분포
    # =====================================================================
    print("\n=== 1. OD 데이터 분포 분석 ===")
    total_elements = num_nodes * num_nodes
    zero_count = np.sum(X_OD == 0)
    non_zero_count = total_elements - zero_count
    
    print(f"- 전체 행렬 크기: {num_nodes} x {num_nodes} = {total_elements:,}")
    print(f"- 0인 값(No Trip) 개수: {zero_count:,} ({zero_count/total_elements*100:.2f}%)")
    print(f"- 0이 아닌(Valid Trip) 개수: {non_zero_count:,} ({non_zero_count/total_elements*100:.2f}%)")
    
    bins = [0, 1, 11, 51, 101, 501, 1001, 5001, 10001, float('inf')]
    labels = ['0', '1~10', '11~50', '51~100', '101~500', '501~1000', '1001~5000', '5001~10000', '10000+']
    
    # Numpy digitize를 이용한 구간 분할
    indices = np.digitize(X_OD.flatten(), bins) - 1
    # 카운트 집계
    val_counts = np.bincount(indices, minlength=len(labels))
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, val_counts, color='skyblue', edgecolor='black')
    plt.title('Distribution of OD Trips by Ranges')
    plt.xlabel('Trip Volume Range')
    plt.ylabel('Count (Log Scale)')
    plt.yscale('log')
    plt.xticks(rotation=45)
    
    # 막대 위에 값 표시
    for bar in bars:
        yval = bar.get_height()
        if yval > 0:
            plt.text(bar.get_x() + bar.get_width()/2, yval, f'{int(yval):,}', 
                     ha='center', va='bottom', fontsize=8)
            
    plt.tight_layout()
    plt.savefig('od_distribution.png')
    print("-> 분포 시각화 완료: 'od_distribution.png' 저장됨")
    
    # =====================================================================
    # 2. 0이 아닌 값들만 놓고 Variance / Mean (Dispersion Index)
    # =====================================================================
    print("\n=== 2. Dispersion Index (Variance / Mean) ===")
    non_zero_od = X_OD[X_OD > 0]
    mean_val = np.mean(non_zero_od)
    var_val = np.var(non_zero_od)
    dispersion = var_val / mean_val
    
    print(f"- Non-zero 통행량 평균(Mean): {mean_val:.2f}")
    print(f"- Non-zero 통행량 분산(Variance): {var_val:.2f}")
    print(f"- Dispersion Index (Var/Mean): {dispersion:.2f}")
    if dispersion > 1:
        print("  -> 분산이 평균보다 압도적으로 큽니다. 전형적인 Over-dispersion(과산포) 및 Heavy-tail 특성을 보입니다.")
        
    # =====================================================================
    # 3. np.log1p(total_trips) 히스토그램
    # =====================================================================
    print("\n=== 3. Log1p 스케일 OD 트립 히스토그램 ===")
    log1p_od = np.log1p(X_OD.flatten())
    
    plt.figure(figsize=(10, 6))
    plt.hist(log1p_od, bins=50, color='coral', edgecolor='black')
    plt.title('Histogram of log1p(OD Trips)')
    plt.xlabel('log1p(Trips)')
    plt.ylabel('Frequency (Log Scale)')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('od_log1p_histogram.png')
    print("-> 히스토그램 시각화 완료: 'od_log1p_histogram.png' 저장됨")
    
    # =====================================================================
    # 4. Moran's I (공간적 자기상관성 분석)
    # =====================================================================
    print("\n=== 4. Moran's I (공간적 자기상관성 분석) ===")
    # 외부 라이브러리(PySAL) 없이 거리 행렬을 통해 인접 행렬(Weights) 구성 후 Moran's I 직접 계산
    
    dist_df = pd.read_csv(DIST_DATA_PATH)
    o_dist = dist_df['O_dong_code'].map(dong2idx).values
    d_dist = dist_df['D_dong_code'].map(dong2idx).values
    v_mask2 = pd.notna(o_dist) & pd.notna(d_dist)
    
    X_dist = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    X_dist[o_dist[v_mask2].astype(int), d_dist[v_mask2].astype(int)] = dist_df['distance'].values[v_mask2]
    
    # 인접 기준: 중심 간 거리 5km 이하
    threshold = 5.0 # dist_data.csv의 단위가 km임
    W = (X_dist < threshold) & (X_dist > 0)
    W = W.astype(float)
    
    # 자기 자신과의 연결(Self-loop)은 Moran's I의 가중치 행렬에서 제외
    np.fill_diagonal(W, 0.0)
    
    # Row-standardize W (행별 가중치 합이 1이 되도록 정규화)
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0 # 0으로 나누기 방지
    W = W / row_sums
    
    # 행정동별 총 발생량(Origin sum) 및 총 도착량(Destination sum)
    trip_gen = X_OD.sum(axis=1)
    trip_attr = X_OD.sum(axis=0)
    
    def calc_morans_i(z, W):
        z_mean = z.mean()
        z_dev = z - z_mean
        denom = np.sum(z_dev**2)
        if denom == 0:
            return 0
        
        # W_ij * z_i * z_j
        num = np.sum(W * np.outer(z_dev, z_dev))
        
        W_sum = W.sum()
        N = len(z)
        I = (N / W_sum) * (num / denom)
        return I

    I_gen = calc_morans_i(trip_gen, W)
    I_attr = calc_morans_i(trip_attr, W)
    
    print(f"- 분석 기준: 인접 행정동을 중심 간 거리 {threshold}km 이내로 정의")
    print(f"- Trip Generation (총 발생량) Moran's I: {I_gen:.4f}")
    print(f"- Trip Attraction (총 도착량) Moran's I: {I_attr:.4f}")
    
    print("  [해석]")
    print("  지표가 0에 가까우면 통행량이 공간적으로 무작위 분포함을 뜻하며,")
    print("  양수(0~1)면 인접한 지역끼리 서로 통행량이 비슷한 현상(공간적 군집화)이 있음을 의미합니다.")
    print("===========================================")

if __name__ == '__main__':
    main()
