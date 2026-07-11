import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정
plt.rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import ODDataset
from model import DoublyConstrainedGravityModel

def cpc_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    numerator = 2 * np.sum(np.minimum(y_true, y_pred))
    denominator = np.sum(y_true) + np.sum(y_pred)
    if denominator == 0: return 0.0
    return numerator / denominator

def analyze():
    print("Loading datasets...")
    dataset = ODDataset() 
    
    # 1. Train 마스크 생성 및 파싱
    train_mask = np.ones(dataset.num_nodes, dtype=bool)
    train_mask[dataset.test_indices] = False
    
    x_od = dataset.X_OD.copy()
    x_od_train = x_od.copy()
    x_od_train[:, ~train_mask] = 0
    x_od_train[~train_mask, :] = 0
    
    O_true_train = np.sum(x_od_train, axis=1)
    D_true_train = np.sum(x_od_train, axis=0)
    y_self_train = np.diag(x_od_train)
    y_inter_train = O_true_train - y_self_train
    
    # 스케일링 보정 (단순 노드 비율)
    scaling_factor = dataset.num_nodes / len(dataset.train_indices)
    O_train_adjusted = O_true_train[train_mask] * scaling_factor
    D_train_adjusted = D_true_train[train_mask] * scaling_factor
    y_self_adjusted = y_self_train[train_mask] * scaling_factor
    y_inter_adjusted = y_inter_train[train_mask] * scaling_factor
    
    X_static_train = dataset.X_static[train_mask]
    
    # 2. 모델 학습 (log1p 적용)
    print("Training Doubly Constrained Gravity Model...")
    model = DoublyConstrainedGravityModel(beta=1.5, max_iter=100)
    model.fit_lgbm_O_D(X_static_train, O_train_adjusted, D_train_adjusted)
    model.fit_lgbm_self_inter(X_static_train, y_self_adjusted, y_inter_adjusted)
    
    # 3. 모델 예측 (expm1 복원)
    O_pred_all, D_pred_all = model.predict_O_D(dataset.X_static)
    y_self_pred_all, y_inter_pred_all = model.predict_self_inter(dataset.X_static)
    
    # 중력모형 배분 (IPF)
    T_pred = model.apply_ipf(O_pred_all, D_pred_all, dataset.X_dist, y_self=y_self_pred_all, y_inter=y_inter_pred_all)
    
    # 4. 성능 평가 데이터 준비
    # 평가 대상 마스크 (Test 노드와 관련된 모든 통행)
    test_mask_2d = np.zeros((dataset.num_nodes, dataset.num_nodes), dtype=bool)
    test_mask_2d[:, dataset.test_indices] = True
    test_mask_2d[dataset.test_indices, :] = True
    
    # 실제 정답값 (전체 매트릭스 기반 총합)
    O_true_all = np.sum(dataset.X_OD, axis=1)
    D_true_all = np.sum(dataset.X_OD, axis=0)
    y_self_true_all = np.diag(dataset.X_OD)
    y_inter_true_all = O_true_all - y_self_true_all
    
    # Test 마스크에 해당하는 정답 및 예측 1D 배열
    y_true_test = dataset.X_OD[test_mask_2d]
    y_pred_test = T_pred[test_mask_2d]
    
    # O, D, Self, Inter 평가 (Test 구역 기준)
    O_true_test = O_true_all[~train_mask]
    O_pred_test = O_pred_all[~train_mask]
    D_true_test = D_true_all[~train_mask]
    D_pred_test = D_pred_all[~train_mask]
    
    # 직접 예측한 LGBM Self, Inter vs Gravity 분배 결과 Self, Inter
    self_true_test = y_self_true_all[~train_mask]
    self_pred_lgbm = y_self_pred_all[~train_mask]
    self_pred_grav = np.diag(T_pred)[~train_mask]
    
    inter_true_test = y_inter_true_all[~train_mask]
    inter_pred_lgbm = y_inter_pred_all[~train_mask]
    inter_pred_grav = np.sum(T_pred, axis=1)[~train_mask] - self_pred_grav
    
    print("\n" + "="*50)
    print(" [요청사항 2,3] LGBM 예측 vs 중력모형 분배 정확도 비교")
    print("="*50)
    metrics = {
        "Origin (LGBM)": (O_true_test, O_pred_test),
        "Destination (LGBM)": (D_true_test, D_pred_test),
        "Self (LGBM 직접예측)": (self_true_test, self_pred_lgbm),
        "Self (Gravity 배분)": (self_true_test, self_pred_grav),
        "Inter (LGBM 직접예측)": (inter_true_test, inter_pred_lgbm),
        "Inter (Gravity 배분)": (inter_true_test, inter_pred_grav)
    }
    
    for name, (true_val, pred_val) in metrics.items():
        rmse = np.sqrt(np.mean((true_val - pred_val)**2))
        cpc = cpc_score(true_val, pred_val)
        print(f"{name:20s} - RMSE: {rmse:7.2f}, CPC: {cpc:.4f}")
        
    print("\n" + "="*50)
    print(" [요청사항 1] 통행량 크기별 정확도 (Gravity Matrix)")
    print("="*50)
    bins = [-1, 0, 10, 50, 200, 1000, np.inf]
    labels = ['0', '1~10', '11~50', '51~200', '201~1000', '1000+']
    bin_indices = pd.cut(y_true_test, bins=bins, labels=labels)
    
    df = pd.DataFrame({
        'True': y_true_test,
        'Gravity': y_pred_test,
        'Bin': bin_indices
    })
    
    results = []
    for b in labels:
        sub = df[df['Bin'] == b]
        if len(sub) == 0: continue
        y_t = sub['True'].values
        p_g = sub['Gravity'].values
        results.append({
            'Bin': b,
            'Count': len(sub),
            'True Mean': np.mean(y_t),
            'Gravity RMSE': np.sqrt(np.mean((y_t - p_g)**2)),
            'Gravity CPC': cpc_score(y_t, p_g)
        })
        
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))
    
    # ---------------------------
    # Visualization
    # ---------------------------
    plt.figure(figsize=(15, 10))
    
    # 1. 구간별 RMSE
    plt.subplot(2, 2, 1)
    sns.barplot(data=res_df, x='Bin', y='Gravity RMSE', color='green', alpha=0.8)
    plt.title('RMSE by True Value Bin (Test Area)')
    plt.xticks(rotation=45)
    
    # 2. 구간별 CPC
    plt.subplot(2, 2, 2)
    sns.barplot(data=res_df, x='Bin', y='Gravity CPC', color='green', alpha=0.8)
    plt.title('CPC by True Value Bin (Test Area)')
    plt.xticks(rotation=45)
    
    # 3. LGBM vs Gravity Self/Inter 산점도
    plt.subplot(2, 2, 3)
    plt.scatter(self_true_test, self_pred_grav, alpha=0.5, label='Self (Gravity)', marker='o')
    plt.scatter(self_true_test, self_pred_lgbm, alpha=0.5, label='Self (LGBM)', marker='x')
    max_val = max(self_true_test.max(), self_pred_grav.max(), self_pred_lgbm.max())
    plt.plot([0, max_val], [0, max_val], 'r--')
    plt.xlabel('True Self')
    plt.ylabel('Predicted Self')
    plt.title('Self Trip: LGBM vs Gravity')
    plt.legend()
    
    # 4. 전체 OD 산점도 (Log scale)
    plt.subplot(2, 2, 4)
    plt.scatter(y_true_test + 1, y_pred_test + 1, alpha=0.05, color='green', s=2)
    plt.plot([1, 100000], [1, 100000], 'r--')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('True Value (+1)')
    plt.ylabel('Predicted Value (+1)')
    plt.title('True vs Predicted OD (Gravity Matrix, Log Scale)')
    
    plt.tight_layout()
    plt.savefig('performance_analysis_gravity.png', dpi=300)
    print("\nSaved visualization to 'performance_analysis_gravity.png'")


if __name__ == "__main__":
    analyze()
