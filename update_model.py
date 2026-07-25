import re

with open('src/gravity(경훈)/model.py', 'r') as f:
    text = f.read()

# 1. DoublyConstrainedGravityModel.__init__
old_init = """    def __init__(self, beta=2.0, max_iter=100, tol=1e-4, generation_model_type='lgbm'):
        \"\"\"
            이중제약 중력모델 초기화

            beta: 마찰계수(거리저항) 지수
            max_iter: IPF 알고리즘 최대 반복 횟수
            tol: IPF 알고리즘 수렴 허용 오차
            generation_model_type: 통행발생량 예측 모델 유형 
                ('lgbm', 'trip_rate', 'cross_class', 'linear_regression')
        \"\"\"
        self.beta = beta
        self.max_iter = max_iter
        self.tol = tol
        self.generation_model_type = generation_model_type
        
        def create_model():
            if generation_model_type == 'trip_rate':
                return TripRateModel()
            elif generation_model_type == 'linear_regression':
                return LinearRegressionModel()
            elif generation_model_type == 'cross_class':
                return CrossClassificationModel()
            else: # lgbm 기본
                return lgb.LGBMRegressor(n_estimators=300, num_leaves=15, min_child_samples=10)
        
        self.model_O = create_model()
        self.model_D = create_model()
        
        # 자기동 내부 통행량 예측도 동일한 모델 유형 사용
        self.model_self = create_model()
        self.model_inter = create_model()"""
        
new_init = """    def __init__(self, generation_model_type='lgbm', beta=2.0, max_iter=100, tol=1e-4):
        self.generation_model_type = generation_model_type
        self.beta = beta
        self.max_iter = max_iter
        self.tol = tol
        
        # 모델 타입 선택
        if self.generation_model_type == 'trip_rate':
            self.model_O = TripRateModel()
            self.model_D = TripRateModel()
        elif self.generation_model_type == 'cross_class':
            self.model_O = CrossClassificationModel()
            self.model_D = CrossClassificationModel()
        elif self.generation_model_type == 'linear_regression':
            self.model_O = LinearRegressionModel()
            self.model_D = LinearRegressionModel()
        elif self.generation_model_type == 'lgbm':
            self.model_O = lgb.LGBMRegressor(
                n_estimators=100, learning_rate=0.05, max_depth=5, min_child_samples=5, random_state=42
            )
            self.model_D = lgb.LGBMRegressor(
                n_estimators=100, learning_rate=0.05, max_depth=5, min_child_samples=5, random_state=42
            )
        else:
            raise ValueError(f"Unknown generation_model_type: {self.generation_model_type}")"""

text = text.replace(old_init, new_init)

# 2. fit_lgbm_O_D
old_fit_od = """    def fit_lgbm_O_D(self, X_static, O_true, D_true):
        \"\"\"
        [1단계] LightGBM을 이용하여 각 동의 통행 발생량(O_i)과 도착량(D_j) 학습
        \"\"\"
        print(f"Training Model for Origin Generation (O_i) using {self.generation_model_type}...")
        use_log = self.generation_model_type not in ('trip_rate',) # 원단위법만 raw scale
        y_O = np.log1p(O_true) if use_log else O_true
        y_D = np.log1p(D_true) if use_log else D_true
        X_s = np.ascontiguousarray(X_static, dtype=np.float64)
        
        self.model_O.fit(X_s, np.ascontiguousarray(y_O, dtype=np.float64))
        
        print(f"Training Model for Destination Attraction (D_j) using {self.generation_model_type}...")
        self.model_D.fit(X_s, np.ascontiguousarray(y_D, dtype=np.float64))"""

new_fit_od = """    def fit_lgbm_O_D(self, X_static, O_true, D_true):
        \"\"\"
        [1단계] 모델을 이용하여 각 동의 통행 발생량(O_i)과 도착량(D_j) 학습
        \"\"\"
        print(f"Training Model for Origin Generation (O_i) using {self.generation_model_type}...")
        use_log = self.generation_model_type not in ('trip_rate', 'cross_class', 'linear_regression')
        
        y_O = np.ascontiguousarray(np.log1p(O_true) if use_log else O_true, dtype=np.float64)
        y_D = np.ascontiguousarray(np.log1p(D_true) if use_log else D_true, dtype=np.float64)
        X_s = np.ascontiguousarray(X_static, dtype=np.float64)

        self.model_O.fit(X_s, y_O)

        print(f"Training Model for Destination Generation (D_j) using {self.generation_model_type}...")
        self.model_D.fit(X_s, y_D)"""
        
text = text.replace(old_fit_od, new_fit_od)

# 3. predict_O_D
old_pred_od = """    def predict_O_D(self, X_static):
        \"\"\"
        학습된 모델로 O_i, D_j 예측
        \"\"\"
        use_log = self.generation_model_type not in ('trip_rate',)
        
        if use_log:
            O_pred = np.maximum(np.expm1(self.model_O.predict(X_static)), 1e-6) # 음수 방지
            D_pred = np.maximum(np.expm1(self.model_D.predict(X_static)), 1e-6)
        else:
            O_pred = np.maximum(self.model_O.predict(X_static), 1e-6)
            D_pred = np.maximum(self.model_D.predict(X_static), 1e-6)

        # 이중제약에서는 총 발생량과 총 도착량이 일치해야 한다길래 스케일링 했어.
        # TODO 이 부분은 필요에 따라 조정 가능. 현재는 총 발생량과 총 도착량을 맞추기 위해 스케일링 적용.
        total_O = np.sum(O_pred)
        total_D = np.sum(D_pred)
        D_pred = D_pred * (total_O / total_D)
        
        return O_pred, D_pred"""
        
new_pred_od = """    def predict_O_D(self, X_static):
        \"\"\"
        [1-2단계] 학습된 발생량/도착량 모델을 사용하여 예측
        \"\"\"
        use_log = self.generation_model_type not in ('trip_rate', 'cross_class', 'linear_regression')
        X_s = np.ascontiguousarray(X_static, dtype=np.float64)
        O_pred = self.model_O.predict(X_s)
        D_pred = self.model_D.predict(X_s)

        if use_log:
            O_pred = np.expm1(O_pred)
            D_pred = np.expm1(D_pred)
            
        O_pred = np.maximum(O_pred, 0)
        D_pred = np.maximum(D_pred, 0)
        return O_pred, D_pred"""
        
text = text.replace(old_pred_od, new_pred_od)

# 4. Remove fit_lgbm_self_inter, predict_self_inter, predict_self
# These are between predict_O_D and apply_ipf. Let's isolate the part to keep.
pattern = r'(    def predict_O_D\(self, X_static\):.*?return O_pred, D_pred)\s+def predict_self_inter.*?    def apply_ipf'
text = re.sub(pattern, r'\1\n\n    def apply_ipf', text, flags=re.DOTALL)

# 5. apply_ipf
old_apply_ipf = """    def apply_ipf(self, O_pred, D_pred, dist_matrix, y_self=None, y_inter=None):
        \"\"\"
        [3단계] Iterative Proportional Fitting 이용

        이 함수는 외부 OD만 배분하는 역할을 한다.

        2026-07-12 수정 설명:
        - 기존 코드에서는 내부통행과 외부통행이 명시적으로 분리되지 않은 상태에서
          거리저항 행렬을 이용해 OD를 배분할 가능성이 있었음.
        - 현재 구조에서는 IPF가 내부통행을 만들지 않도록 대각선을 막고,
          내부통행은 별도 모델 예측값(y_self)으로 마지막에 대각선에 넣음.

        전제:
        - O_pred는 전체 발생량이 아니라 내부통행을 제외한 외부 유출 총량이어야 함.
        - D_pred는 전체 도착량이 아니라 내부통행을 제외한 외부 유입 총량이어야 함.
        - y_self는 최종 OD 대각선에 들어갈 내부통행량임.

        예:
        - A동 전체 발생량 = 1500
        - A동 내부통행 = 1000
        - A동 외부유출 = 500
        이 구조에서는 IPF에 O_pred=1500이 아니라 O_pred=500이 들어가야 함.
        O_pred=1500을 넣고 마지막에 내부통행 1000을 또 넣으면 중복 계산됨.

        최종 OD 구조:
        - 대각선 제외: IPF로 배분한 외부 OD
        - 대각선: y_self
        \"\"\"
        # 총 동 개수
        num_nodes = len(O_pred)
        # numpy 배열로 변환
        O_pred = np.asarray(O_pred, dtype=np.float64).copy()
        D_pred = np.asarray(D_pred, dtype=np.float64).copy()
        dist_matrix = np.asarray(dist_matrix, dtype=np.float64)

        if dist_matrix.shape != (num_nodes, num_nodes):
            raise ValueError(
                "dist_matrix shape must match O_pred/D_pred length: "
                f"{dist_matrix.shape} != ({num_nodes}, {num_nodes})"
            )

        if y_self is not None:
            y_self = np.asarray(y_self, dtype=np.float64)
            if y_self.shape != (num_nodes,):
                raise ValueError(
                    "y_self length must match O_pred/D_pred length: "
                    f"{y_self.shape} != ({num_nodes},)"
                )

        # 음수나 0 방지
        O_pred = np.maximum(O_pred, 1e-9)
        D_pred = np.maximum(D_pred, 1e-9)

        # IPF는 전체 외부 유출합과 전체 외부 유입합이 같아야 함
        total_O = np.sum(O_pred)
        total_D = np.sum(D_pred)

        if total_O <= 0 or total_D <= 0:
            raise ValueError("O_pred와 D_pred의 총합은 0보다 커야 함.")

        # 예측값 차이 때문에 총합이 다를 수 있으므로 D를 O 총합에 맞춤
        # 단, 이건 단순 예측오차/반올림 차이를 보정하는 용도
        D_pred = D_pred * (total_O / total_D)
        
        # 거리저항 계산.
        # 거리가 가까울수록 f_d가 커지고, 멀수록 f_d가 작아진다.
        # 현재 식: f(d_ij) = 1 / d_ij^beta
        dist_safe = np.maximum(dist_matrix, 1e-3)
        f_d = dist_safe ** (-self.beta)

        # 중요: 외부 OD 배분에서는 대각선을 반드시 0으로 둔다.
        # 원래 거리 0인 대각선은 dist_safe에서 1e-3으로 바뀐다.
        # beta=1.5일 때 1 / 0.001^1.5 ~= 31623이 되어
        # 자기동(A동->A동)에 지나치게 큰 가중치가 생길 수 있다.
        # 하지만 현재 구조에서는 내부통행을 y_self로 따로 예측해서
        # 마지막에 대각선에 넣으므로, IPF 단계에서는 자기동 배분을 막는다.(그러니까 A동 내부 통행량을 외부 통행량으로 옮기는거 방지)
        np.fill_diagonal(f_d, 0.0)
        
        # A_i, B_j 초기화
        A = np.ones(num_nodes)
        B = np.ones(num_nodes)
        
        print("Starting Iterative Proportional Fitting (IPF)...")
        for iteration in range(self.max_iter):
            # 행 합을 O_pred에 맞추기 위한 보정계수 A_i 계산.
            # 각 출발동 i에서 외부로 나가는 총합이 O_pred[i]가 되도록 맞춘다.
            A_new = 1.0 / np.maximum(
                np.sum(B[None, :] * D_pred[None, :] * f_d, axis=1),
                1e-12
            )

            # 열 합을 D_pred에 맞추기 위한 보정계수 B_j 계산.
            # 각 도착동 j로 외부에서 들어오는 총합이 D_pred[j]가 되도록 맞춘다.
            B_new = 1.0 / np.maximum(
                np.sum(A_new[:, None] * O_pred[:, None] * f_d, axis=0),
                1e-12
            )
            
            # 수렴 확인.
            # 기존 AI 코드에서는 A, B 값 변화량으로 수렴을 판단했다.
            # 현재는 실제로 만들어진 OD 행렬의 행합/열합이
            # O_pred/D_pred와 맞는지를 직접 확인한다. (이중제약 중력모델 특성상 이게 더 직관적임)
            A = A_new
            B = B_new

            T_check = (
                A[:, None]
                * O_pred[:, None]
                * B[None, :]
                * D_pred[None, :]
                * f_d
            )
            row_rel_error = np.max(
                np.abs(T_check.sum(axis=1) - O_pred)
                / np.maximum(O_pred, 1e-12)
            )
            col_rel_error = np.max(
                np.abs(T_check.sum(axis=0) - D_pred)
                / np.maximum(D_pred, 1e-12)
            )

            if max(row_rel_error, col_rel_error) < self.tol:
                print(f"IPF converged at iteration {iteration+1}")
                break
                

        T_ext = (
            A[:, None]
            * O_pred[:, None]
            * B[None, :]
            * D_pred[None, :]
            * f_d
        )

        # 외부 OD 행렬이므로 대각선은 다시 한 번 0으로 고정한다.
        np.fill_diagonal(T_ext, 0.0)

        # 최종 확인:
        # - row_error: 각 출발동 외부 유출 총합이 O_pred와 얼마나 다른지
        # - col_error: 각 도착동 외부 유입 총합이 D_pred와 얼마나 다른지
        row_error = np.max(
            np.abs(T_ext.sum(axis=1) - O_pred)
        )
        col_error = np.max(
            np.abs(T_ext.sum(axis=0) - D_pred)
        )
        row_rel_error = np.max(
            np.abs(T_ext.sum(axis=1) - O_pred)
            / np.maximum(O_pred, 1e-12)
        )
        col_rel_error = np.max(
            np.abs(T_ext.sum(axis=0) - D_pred)
            / np.maximum(D_pred, 1e-12)
        )

        print(
            f"IPF row error: {row_error:.6f}, "
            f"relative: {row_rel_error:.6f}"
        )
        print(
            f"IPF col error: {col_error:.6f}, "
            f"relative: {col_rel_error:.6f}"
        )

        # 내부통행 예측값이 있으면 최종 OD 대각선에 삽입한다.
        # 이 단계에서야 A동->A동 같은 자기동 통행이 들어간다.
        if y_self is not None:
            T_final = T_ext.copy()
            np.fill_diagonal(T_final, y_self)
            return T_final

        return T_ext"""
        
new_apply_ipf = """    def apply_ipf(self, O_pred, D_pred, dist_matrix):
        \"\"\"
        [3단계] IPF (Iterative Proportional Fitting)를 통해 OD 행렬 갱신.
        \"\"\"
        num_nodes = len(O_pred)
        O_pred = np.asarray(O_pred, dtype=np.float64).copy()
        D_pred = np.asarray(D_pred, dtype=np.float64).copy()
        dist_matrix = np.asarray(dist_matrix, dtype=np.float64)

        if dist_matrix.shape != (num_nodes, num_nodes):
            raise ValueError(f"dist_matrix shape must match: {dist_matrix.shape}")

        O_pred = np.maximum(O_pred, 1e-9)
        D_pred = np.maximum(D_pred, 1e-9)

        total_O = np.sum(O_pred)
        total_D = np.sum(D_pred)
        D_pred = D_pred * (total_O / total_D)
        
        dist_safe = np.maximum(dist_matrix, 1e-3)
        f_d = dist_safe ** (-self.beta)

        A = np.ones(num_nodes)
        B = np.ones(num_nodes)
        
        print("Starting Iterative Proportional Fitting (IPF)...")
        for iteration in range(self.max_iter):
            A_new = 1.0 / np.maximum(
                np.sum(B[None, :] * D_pred[None, :] * f_d, axis=1),
                1e-12
            )

            B_new = 1.0 / np.maximum(
                np.sum(A_new[:, None] * O_pred[:, None] * f_d, axis=0),
                1e-12
            )
            
            A = A_new
            B = B_new

            T_check = (
                A[:, None]
                * O_pred[:, None]
                * B[None, :]
                * D_pred[None, :]
                * f_d
            )
            row_rel_error = np.max(
                np.abs(T_check.sum(axis=1) - O_pred)
                / np.maximum(O_pred, 1e-12)
            )
            col_rel_error = np.max(
                np.abs(T_check.sum(axis=0) - D_pred)
                / np.maximum(D_pred, 1e-12)
            )

            if max(row_rel_error, col_rel_error) < self.tol:
                print(f"IPF converged at iteration {iteration+1}")
                break

        T_ext = (
            A[:, None]
            * O_pred[:, None]
            * B[None, :]
            * D_pred[None, :]
            * f_d
        )
        
        # 외부 OD 행렬이므로 대각선은 다시 한 번 0으로 고정한다.
        # np.fill_diagonal(T_ext, 0.0)
        # 이제 대각선에 바로 intrazonal distance가 들어가므로 0으로 고정하지 않는다.

        row_error = np.max(np.abs(T_ext.sum(axis=1) - O_pred))
        col_error = np.max(np.abs(T_ext.sum(axis=0) - D_pred))

        print(
            f"IPF row error: {row_error:.6f}, "
            f"col error: {col_error:.6f}"
        )

        return T_ext"""
        
text = text.replace(old_apply_ipf, new_apply_ipf)

# 6. fit_predict
old_fit_predict = """    def fit_predict(self, X_static_train, O_train, D_train, X_static_all, dist_matrix, X_self=None, X_inter=None):
        \"\"\"
        [4단계] 전체 파이프라인 통합 실행.

        2026-07-12 수정 설명:
        - O_train/D_train은 전체 통행량이 아니라 외부 유출/외부 유입 총량이어야 함.
          이 값은 train_and_test.py에서 대각선 내부통행을 뺀 뒤 만들어짐.
        - X_self는 이름은 X로 되어 있지만 실제로는 내부통행 정답값(y_self_train)에 가까움.
        - 내부통행 모델을 학습한 뒤 predict_self()로 전체 동의 내부통행을 예측하고
          apply_ipf() 마지막 단계에서 최종 OD 대각선에 넣음.
        \"\"\"
        # 총 발생량, 총 도착량 예측용 LGBM 학습
        self.fit_lgbm_O_D(X_static_train, O_train, D_train)
        O_pred, D_pred = self.predict_O_D(X_static_all)
        
        # 자기동 내부 통행량, 타 지역 간 통행량 예측용 LGBM 학습 (선택 사항)
        y_self_pred = None
        if X_self is not None:
            self.fit_lgbm_self_inter(X_static_train, X_self)
            y_self_pred = self.predict_self(X_static_all)
        
        T_pred = self.apply_ipf(O_pred, D_pred, dist_matrix, y_self=y_self_pred, y_inter=None)
        return T_pred"""

new_fit_predict = """    def fit_predict(self, X_static_train, O_train, D_train, X_static_all, dist_matrix):
        \"\"\"
        [4단계] 전체 파이프라인 통합 실행.
        \"\"\"
        # 총 발생량, 총 도착량 예측용 모델 학습
        self.fit_lgbm_O_D(X_static_train, O_train, D_train)
        O_pred, D_pred = self.predict_O_D(X_static_all)
        
        # 거리 매트릭스 대각선을 최소 인접 거리 / 2.0 으로 설정
        dist_no_diag = dist_matrix.copy()
        np.fill_diagonal(dist_no_diag, np.inf)
        intrazonal_dist = dist_no_diag.min(axis=1) / 2.0
        
        dist_matrix_modified = dist_matrix.copy()
        np.fill_diagonal(dist_matrix_modified, intrazonal_dist)
        
        T_pred = self.apply_ipf(O_pred, D_pred, dist_matrix_modified)
        return T_pred"""

text = text.replace(old_fit_predict, new_fit_predict)

# Also fix the inner function in fit_predict if it wasn't matched properly.
# Actually, I'll just write it.
with open('src/gravity(경훈)/model.py', 'w') as f:
    f.write(text)

print("model.py successfully updated!")
