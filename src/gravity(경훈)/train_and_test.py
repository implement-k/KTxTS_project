import os
os.environ['KMP_DUPLICATE_OK'] = 'True'
import numpy as np
from dataset import ODDataset
from model import DoublyConstrainedGravityModel

'''
    이 코드는 수정할 필요 없을거야. models.py에서 코드 수정하면 돼.
    이파일 실행하면 모델 학습하고, Test 구역(동탄, 위례, 검단)에 대한 RMSE, CPC 평가 결과 출력해줘.
'''


def main():
    dataset = ODDataset()
    
    # 1. Train 노드 마스킹 (동탄, 위례, 검단)
    train_mask = np.ones(dataset.num_nodes, dtype=bool)
    train_mask[dataset.test_indices] = False
    
    # 2. 데이터 파싱
    # O_i:발생량, D_j:도착량
    x_od = dataset.X_OD.copy()
    x_od[:, ~train_mask] = 0 # Test 도착 가리기
    x_od[~train_mask, :] = 0 # Test 출발 가리기
    
    # 2.1. 총 발생량, 총 도착량, 자기동 내부 통행량, 타 지역 간 통행량 계산
    '''
        y_self랑 y_inter는 필요하면 써
    '''
    y_o = np.sum(x_od, axis=1) # 행 합계 (발생량, Origin) (N,)
    y_d = np.sum(x_od, axis=0) # 열 합계 (도착량, Destination) (N,)
    y_self = np.diag(x_od) # 자기동 내부 통행량 (N,)
    y_inter = y_o - y_self # 타 지역 간 통행량 (N,)
    
    # 2.2. 학습용 데이터셋 생성
    X_static = dataset.X_static[train_mask]
    X_o = y_o[train_mask]
    X_d = y_d[train_mask]
    X_self = y_self[train_mask]
    X_inter = y_inter[train_mask]
    
    # 3. 모델 초기화 및 실행
    '''
        beta: 마찰계수 지수, max_iter: IPF 최대 반복 횟수
        마찰계수 지수는 일반적으로 1~2 값이라는데 조정하면서 성능이 가장 좋게 나오도록 설정해
    '''
    model = DoublyConstrainedGravityModel(beta=2.0, max_iter=100)
    
    # 4. LGBM 학습 및 이중제약 적용
    # 주의: IPF 알고리즘은 전체 노드에 대해 수행되어야 하므로 X_dist가 아닌 전체 dataset.X_dist를 전달해야 합니다.
    T_pred = model.fit_predict(X_static, X_o, X_d, dataset.X_static, dataset.X_dist, X_self, X_inter)
    
    print("Shape of Predicted OD Matrix:", T_pred.shape)
    
    # 5. Test 데이터 셋에 대한 RMSE, CPC 평가 로직 추가
    y_od_all = dataset.X_OD.copy()
    
    # 5.1. 평가 대상 마스크 (Test 노드로 가거나 Test 노드에서 오는 모든 통행)
    test_mask_2d = np.zeros((dataset.num_nodes, dataset.num_nodes), dtype=bool)
    test_mask_2d[:, dataset.test_indices] = True
    test_mask_2d[dataset.test_indices, :] = True
    
    # 5.2. test 도시 추출 (검단, 위례, 동탄)
    y_od_test = y_od_all[test_mask_2d]
    y_pred_test = T_pred[test_mask_2d]

    # 전체 평가 (모든 동)
    rmse_all = np.sqrt(np.mean((y_od_all - T_pred)**2))
    cpc_all = cpc_score(y_od_all, T_pred)
    
    # 마스킹된 Test 영역 평가 (동탄, 위례, 검단)
    rmse_test = np.sqrt(np.mean((y_od_test - y_pred_test)**2))
    cpc_test = cpc_score(y_od_test, y_pred_test)
    
    print("\n" + "="*30)
    print("    === 평가 결과 ===")
    print(f"[전체 OD 매트릭스 (모든동)]")
    print(f" - RMSE: {rmse_all:.4f}")
    print(f" - CPC : {cpc_all:.4f}\n")
    
    print(f"[마스킹된 Test 구역 (동탄, 위례, 검단)]")
    print(f" - RMSE: {rmse_test:.4f}")
    print(f" - CPC : {cpc_test:.4f}")
    print("="*30)

# CPC 점수 함수
def cpc_score(y_t, y_p):
    num = 2 * np.sum(np.minimum(y_t, y_p))
    den = np.sum(y_t) + np.sum(y_p)
    return num / den if den > 0 else 0.0

if __name__ == "__main__":
    main()
