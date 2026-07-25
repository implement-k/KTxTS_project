import os
os.environ['KMP_DUPLICATE_OK'] = 'True'
import numpy as np
import argparse
from dataset import ODDataset
from model import DoublyConstrainedGravityModel

def main():
    '''
        실행방법: python train_and_test.py --model_type [lgbm|trip_rate|cross_class|linear_regression] --imputation [zero|mean]
        model_type: 통행발생량 예측 모델 유형 선택
        imputation: 결측치(마스킹된 피처) 처리 방식 (zero: 0.0, mean: 학습 데이터 평균)
    '''
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='lgbm',
                        choices=['lgbm', 'trip_rate', 'cross_class', 'linear_regression'],
                        help='통행발생량 예측 모델 유형 선택')
    parser.add_argument('--imputation', type=str, default='zero',
                        choices=['zero', 'mean'],
                        help='결측치(마스킹된 피처) 처리 방식 (zero: 0.0, mean: 학습 데이터 평균)')
    args = parser.parse_args()
    
    dataset_2019 = ODDataset(year=2019, imputation=args.imputation)
    dataset_2023 = ODDataset(year=2023, imputation=args.imputation)
    
    # Validation uses the 2023 dataset layout (number of nodes and masks should match)
    dataset = dataset_2023
    
    # 1. Train 노드 마스킹 (일산, 판교, 평택 등)
    train_mask = np.ones(dataset.num_nodes, dtype=bool)
    test_indices = []
    
    from config import VAL_CITIES_CODES
    for _, nodes in VAL_CITIES_CODES.items():
        for n in nodes:
            idx = np.where(dataset.dong_codes == int(n))[0]
            if len(idx) > 0:
                train_mask[idx[0]] = False
                test_indices.append(idx[0])
    
    # 2. 데이터 파싱 (2019, 2023)
    def parse_od(ds):
        x = ds.X_OD.copy()
        x[:, ~train_mask] = 0
        x[~train_mask, :] = 0
        y_o = np.sum(x, axis=1)
        y_d = np.sum(x, axis=0)
        return y_o, y_d
        
    y_o_19, y_d_19 = parse_od(dataset_2019)
    y_o_23, y_d_23 = parse_od(dataset_2023)
    
    # 2.2. 학습용 데이터셋 생성 및 병합
    if args.model_type in ('trip_rate', 'cross_class', 'linear_regression'):
        X_static_19 = dataset_2019.X_static_raw[train_mask]
        X_static_23 = dataset_2023.X_static_raw[train_mask]
        X_static_all = dataset_2023.X_static_raw
    else:
        X_static_19 = dataset_2019.X_static[train_mask]
        X_static_23 = dataset_2023.X_static[train_mask]
        X_static_all = dataset_2023.X_static
        
    X_static_train = np.concatenate([X_static_19, X_static_23], axis=0)
    X_o_train = np.concatenate([y_o_19[train_mask], y_o_23[train_mask]], axis=0)
    X_d_train = np.concatenate([y_d_19[train_mask], y_d_23[train_mask]], axis=0)

    # 3. 모델 초기화 및 실행
    '''
        beta: 마찰계수 지수, max_iter: IPF 최대 반복 횟수
        마찰계수 지수는 일반적으로 1~2 값이라는데 조정하면서 성능이 가장 좋게 나오도록 설정해
    '''
    print(f"Initializing DoublyConstrainedGravityModel with generation_model_type='{args.model_type}'")
    model = DoublyConstrainedGravityModel(beta=2.0, max_iter=100, tol=1e-4, generation_model_type=args.model_type)
    
    # 4. LGBM 학습 및 이중제약 적용
    # 주의: IPF 알고리즘은 전체 노드에 대해 수행되어야 하므로 X_dist가 아닌 전체 dataset.X_dist를 전달해야함
    print("Start Model Training and IPF prediction...")
    T_pred = model.fit_predict(X_static_train, X_o_train, X_d_train, X_static_all, np.expm1(dataset.X_dist))
    
    ##### 주의! 주의! 주의! #####
    ##### 절대로 test 결과를 보고 모델을 수정하거나, 튜닝하면 안됨. 기준은 val을 보고 결정해야함! ######
    print("Shape of Predicted OD Matrix:", T_pred.shape)
    
    # 5. Test 데이터 셋에 대한 RMSE, CPC 평가 로직 추가
    y_od_all = dataset.X_OD.copy()
    
    # 5.1. 평가 대상 마스크 (Test 노드로 가거나 Test 노드에서 오는 모든 통행)
    test_mask_2d = np.zeros((dataset.num_nodes, dataset.num_nodes), dtype=bool)
    test_mask_2d[:, test_indices] = True
    test_mask_2d[test_indices, :] = True
    
    # 5.2. test 도시 추출 (검단, 위례, 동탄)
    y_od_test = y_od_all[test_mask_2d]
    y_pred_test = T_pred[test_mask_2d]

    # 전체 평가 (모든 동)
    rmse_all = np.sqrt(np.mean((y_od_all - T_pred)**2))
    cpc_all = cpc_score(y_od_all, T_pred)
    
    # 마스킹된 Test 영역 평가 (동탄, 위례, 검단 등)
    rmse_test = np.sqrt(np.mean((y_od_test - y_pred_test)**2))
    cpc_test = cpc_score(y_od_test, y_pred_test)
    
    prmse_all = rmse_all / np.mean(y_od_all) if np.mean(y_od_all) > 0 else 0.0
    prmse_test = rmse_test / np.mean(y_od_test) if np.mean(y_od_test) > 0 else 0.0
    
    print("\n" + "="*30)
    print("    === 평가 결과 ===")
    print(f"[전체 OD 매트릭스 (모든동)]")
    print(f" - RMSE  : {rmse_all:.4f}")
    print(f" - CPC   : {cpc_all:.4f}")
    print(f" - %RMSE : {prmse_all:.4f}\n")
    
    print("\n[마스킹된 Validation 구역]")
    print(f" - RMSE  : {rmse_test:.4f}")
    print(f" - CPC   : {cpc_test:.4f}")
    print(f" - %RMSE : {prmse_test:.4f}")
    print("="*30)

# CPC 점수 함수
def cpc_score(y_t, y_p):
    num = 2 * np.sum(np.minimum(y_t, y_p))
    den = np.sum(y_t) + np.sum(y_p)
    return num / den if den > 0 else 0.0

if __name__ == "__main__":
    main()
