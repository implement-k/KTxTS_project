import os
os.environ['KMP_DUPLICATE_OK'] = 'True'
import numpy as np
import lightgbm as lgb

'''
    여기 코드 수정해주면 돼.
'''


class DoublyConstrainedGravityModel:
    def __init__(self, beta=1.5, max_iter=100, tol=1e-4):
        """
            LightGBM + 이중제약 모델 코드 초기화
            
            TODO 아래 매개변수는 필요에 따라 지워도 됨. [AI 내용. 검증 안함]
                beta: 거리 마찰계수 (d_ij ^ -beta)
                max_iter: IPF(Balancing) 최대 반복 횟수
                tol: 수렴 허용 오차
        """
        self.beta = beta
        self.max_iter = max_iter
        self.tol = tol
        
        # O_i (발생량), D_j (도착량) 예측용 LGBM 모델
        self.model_O = lgb.LGBMRegressor(n_estimators=300, num_leaves=15, min_child_samples=10)
        self.model_D = lgb.LGBMRegressor(n_estimators=300, num_leaves=15, min_child_samples=10)
        
        # 자기동 내부 통행량, 타 지역 간 통행량 예측용 LGBM 모델
        self.model_self = lgb.LGBMRegressor(n_estimators=300, num_leaves=15, min_child_samples=10)
        self.model_inter = lgb.LGBMRegressor(n_estimators=300, num_leaves=15, min_child_samples=10)

    def fit_lgbm_O_D(self, X_static, O_true, D_true):
        """
        [1단계] LightGBM을 이용하여 각 동의 통행 발생량(O_i)과 도착량(D_j) 학습
        """
        print("Training LightGBM for Origin Generation (O_i)...")
        y_O = np.ascontiguousarray(np.log1p(O_true), dtype=np.float64)
        X_s = np.ascontiguousarray(X_static, dtype=np.float64)
        self.model_O.fit(X_s, y_O)
        
        print("Training LightGBM for Destination Attraction (D_j)...")
        y_D = np.ascontiguousarray(np.log1p(D_true), dtype=np.float64)
        self.model_D.fit(X_s, y_D)

    def fit_lgbm_self_inter(self, X_static, y_self, y_inter):
        """
        [1단계] LightGBM을 이용하여 자기동 내부 통행량(y_self)과 타 지역 간 통행량(y_inter) 학습
        선택 사항 필요하면 써.
        """
        print("Training LightGBM for Self-Trip Prediction (y_self)...")
        y_s = np.ascontiguousarray(np.log1p(y_self), dtype=np.float64)
        X_s = np.ascontiguousarray(X_static, dtype=np.float64)
        self.model_self.fit(X_s, y_s)
        
        print("Training LightGBM for Inter-Trip Prediction (y_inter)...")
        y_i = np.ascontiguousarray(np.log1p(y_inter), dtype=np.float64)
        self.model_inter.fit(X_s, y_i)

    def predict_O_D(self, X_static):
        """
        학습된 LGBM으로 O_i, D_j 예측
        """
        O_pred = np.maximum(np.expm1(self.model_O.predict(X_static)), 1e-6) # 음수 방지
        D_pred = np.maximum(np.expm1(self.model_D.predict(X_static)), 1e-6)

        # 이중제약에서는 총 발생량과 총 도착량이 일치해야 한다길래 스케일링 했어.
        # TODO 이 부분은 필요에 따라 조정 가능. 현재는 총 발생량과 총 도착량을 맞추기 위해 스케일링 적용.
        total_O = np.sum(O_pred)
        total_D = np.sum(D_pred)
        D_pred = D_pred * (total_O / total_D)
        
        return O_pred, D_pred

    def predict_self_inter(self, X_static):
        """
        학습된 LGBM으로 자기동 내부 통행량(y_self)과 타 지역 간 통행량(y_inter) 예측
        선택 사항 필요하면 써.
        """
        y_self_pred = np.maximum(np.expm1(self.model_self.predict(X_static)), 1e-6)
        y_inter_pred = np.maximum(np.expm1(self.model_inter.predict(X_static)), 1e-6)
        
        return y_self_pred, y_inter_pred
    
    def apply_ipf(self, O_pred, D_pred, dist_matrix, y_self=None, y_inter=None):
        """
        [2단계] Iterative Proportional Fitting 이용
        TODO 이 부분 중점적으로 보고 코드 수정해줘. 지금은 AI가 짠 코드 그대로 있어.
        O_pred: 예측된 발생량 (N,)
        D_pred: 예측된 도착량 (N,)
        dist_matrix: 거리 행렬 (N, N)
        
        아래는 필요하면 써.
        y_self: 자기동 내부 통행량 (N,)
        y_inter: 타 지역 간 통행량 (N,)
        """
        # 총 동 개수
        num_nodes = len(O_pred)
        
        # 마찰 계수 f(d_ij) 계산 (대각선 자가통행 처리는 필요시 수정)
        # distance가 0인 경우 방지를 위해 epsilon 추가
        dist_safe = np.maximum(dist_matrix, 1e-3)
        f_d = dist_safe ** (-self.beta)
        
        # A_i, B_j 초기화
        A = np.ones(num_nodes)
        B = np.ones(num_nodes)
        
        print("Starting Iterative Proportional Fitting (IPF)...")
        for iteration in range(self.max_iter):
            A_new = 1.0 / np.maximum(np.sum(B * D_pred * f_d, axis=1), 1e-12)
            B_new = 1.0 / np.maximum(np.sum(A_new[:, None] * O_pred[:, None] * f_d, axis=0), 1e-12)
            
            # 수렴 확인
            diff_A = np.max(np.abs(A_new - A))
            diff_B = np.max(np.abs(B_new - B))
            
            A = A_new
            B = B_new
            
            if max(diff_A, diff_B) < self.tol:
                print(f"IPF converged at iteration {iteration+1}")
                break
                
        # 최종 통행량 매트릭스 계산: T_ij = A_i * O_i * B_j * D_j * f(d_ij)
        T_ij = A[:, None] * O_pred[:, None] * B[None, :] * D_pred[None, :] * f_d
        return T_ij

    def fit_predict(self, X_static_train, O_train, D_train, X_static_all, dist_matrix, X_self=None, X_inter=None):
        """
        전체 파이프라인 통합 실행
        """
        # 총 발생량, 총 도착량 예측용 LGBM 학습
        self.fit_lgbm_O_D(X_static_train, O_train, D_train)
        O_pred, D_pred = self.predict_O_D(X_static_all)
        
        # 자기동 내부 통행량, 타 지역 간 통행량 예측용 LGBM 학습 (선택 사항)
        y_self_pred, y_inter_pred = None, None
        # self.fit_lgbm_self_inter(X_static_train, X_self, X_inter)
        # y_self_pred, y_inter_pred = self.predict_self_inter(X_static_all)
        
        T_pred = self.apply_ipf(O_pred, D_pred, dist_matrix, y_self=y_self_pred, y_inter=y_inter_pred)
        return T_pred
