import os
os.environ['KMP_DUPLICATE_OK'] = 'True'
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LinearRegression

class TripRateModel:
    """원단위법 (Trip Rate): 상수항 없이 고정 비율(계수)을 피처에 곱하여 통행량을 예측"""
    def __init__(self):
        self.model = LinearRegression(fit_intercept=False)
        
    def fit(self, X, y):
        self.model.fit(X, y)
        
    def predict(self, X)-> np.ndarray:
        return self.model.predict(X)

class LinearRegressionModel:
    """선형회귀분석 (Linear Regression): 상수항을 포함하여 다중 선형회귀식 적용"""
    def __init__(self):
        self.model = LinearRegression(fit_intercept=True)
        
    def fit(self, X, y):
        self.model.fit(X, y)
        
    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

class CrossClassificationModel:
    """
    교차분류분석 (Cross-classification): 
    주요 피처를 기반으로 데이터를 N개의 구간(Bin)으로 나누고, 
    각 Bin 조합(Cell)별 평균 통행발생량을 학습하여 예측.
    """
    def __init__(self, num_features_to_use=3, bins_per_feature=3):
        self.num_features_to_use = num_features_to_use
        self.bins_per_feature = bins_per_feature
        self.cell_means = {}
        self.global_mean = 0.0
        self.bin_edges = []
        
    def fit(self, X, y):
        self.global_mean = np.mean(y)
        # 차원의 저주를 피하기 위해 상위 K개의 피처만 사용
        n_feats = min(self.num_features_to_use, X.shape[1])
        X_used = X[:, :n_feats]
        
        self.bin_edges = []
        X_binned = np.zeros_like(X_used, dtype=int)
        
        for i in range(n_feats):
            # Equal width binning
            min_v, max_v = np.min(X_used[:, i]), np.max(X_used[:, i])
            # min_v와 max_v가 같을 경우 방어 코드
            if min_v == max_v:
                edges = np.array([-np.inf, np.inf])
            else:
                edges = np.linspace(min_v, max_v, self.bins_per_feature + 1)
                edges[0] = -np.inf
                edges[-1] = np.inf
            self.bin_edges.append(edges)
            X_binned[:, i] = np.digitize(X_used[:, i], edges) - 1
            
        from collections import defaultdict
        sums = defaultdict(float)
        counts = defaultdict(int)
        
        for i in range(len(y)):
            cell = tuple(X_binned[i])
            sums[cell] += y[i]
            counts[cell] += 1
            
        self.cell_means = {cell: sums[cell]/counts[cell] for cell in sums}
        
    def predict(self, X)-> np.ndarray:
        n_feats = min(self.num_features_to_use, X.shape[1])
        X_used = X[:, :n_feats]
        preds = []
        for i in range(len(X)):
            cell = []
            for j in range(n_feats):
                c = np.digitize(X_used[i, j], self.bin_edges[j]) - 1
                cell.append(c)
            preds.append(self.cell_means.get(tuple(cell), self.global_mean))
        return np.array(preds)

class DoublyConstrainedGravityModel:
    def __init__(self, generation_model_type='lgbm', beta=2.0, max_iter=10, tol=1e-4): # 임시로 100->10번 TODO 
        self.generation_model_type = generation_model_type
        self.beta = beta
        self.max_iter = max_iter
        self.tol = tol
        
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
                n_estimators=100, learning_rate=0.05, max_depth=5, min_child_samples=5, random_state=42, n_jobs=1
            )
            self.model_D = lgb.LGBMRegressor(
                n_estimators=100, learning_rate=0.05, max_depth=5, min_child_samples=5, random_state=42, n_jobs=1
            )
        else:
            raise ValueError(f"Unknown generation_model_type: {self.generation_model_type}")

    def fit_O_D(self, X_static, O_true, D_true, useLog):
        print(f"Training Model for Origin Generation (O_i) using {self.generation_model_type}...")
        
        y_O = np.ascontiguousarray(np.log1p(O_true) if useLog else O_true, dtype=np.float64)
        y_D = np.ascontiguousarray(np.log1p(D_true) if useLog else D_true, dtype=np.float64)
        X_s = np.ascontiguousarray(X_static, dtype=np.float64)

        # Numerical guard only: generation models can extrapolate outside the
        # physically meaningful range, especially no-intercept trip-rate models.
        # Keep a broad cap from the training totals so expm1/IPF does not turn
        # invalid predictions into inf/nan. This is not tuned from test metrics.
        finite_totals = np.concatenate([
            np.asarray(O_true, dtype=np.float64).reshape(-1),
            np.asarray(D_true, dtype=np.float64).reshape(-1),
        ])
        finite_totals = finite_totals[np.isfinite(finite_totals)]
        self.prediction_cap_ = max(float(np.sum(finite_totals)), 1.0)
        self.log_prediction_cap_ = np.log1p(self.prediction_cap_)

        self.model_O.fit(X_s, y_O)
        self.model_D.fit(X_s, y_D)

    def predict_O_D(self, X_static, useLog):
        X_s = np.ascontiguousarray(X_static, dtype=np.float64)
        O_pred: np.ndarray = self.model_O.predict(X_s) 
        D_pred: np.ndarray = self.model_D.predict(X_s)

        cap = getattr(self, 'prediction_cap_', 1e12)
        if useLog:
            # Clip in log space before expm1. Without this, a few extrapolated
            # validation samples can overflow to inf and poison the IPF matrix.
            log_cap = getattr(self, 'log_prediction_cap_', np.log1p(cap))
            O_pred = np.expm1(np.clip(O_pred, -50.0, log_cap))
            D_pred = np.expm1(np.clip(D_pred, -50.0, log_cap))
        else:
            O_pred = np.maximum(O_pred, 0)
            D_pred = np.maximum(D_pred, 0)

        O_pred = np.nan_to_num(O_pred, nan=0.0, posinf=cap, neginf=0.0)
        D_pred = np.nan_to_num(D_pred, nan=0.0, posinf=cap, neginf=0.0)
        O_pred = np.clip(O_pred, 0.0, cap)
        D_pred = np.clip(D_pred, 0.0, cap)
        return O_pred, D_pred

    def apply_ipf(self, O_pred, D_pred, dist_matrix):
        num_nodes = len(O_pred)
        O_pred = np.asarray(O_pred, dtype=np.float64).copy()
        D_pred = np.asarray(D_pred, dtype=np.float64).copy()
        dist_matrix = np.asarray(dist_matrix, dtype=np.float64)

        if dist_matrix.shape != (num_nodes, num_nodes):
            raise ValueError(f"dist_matrix shape must match: {dist_matrix.shape}")

        cap = getattr(self, 'prediction_cap_', 1e12)
        O_pred = np.nan_to_num(O_pred, nan=0.0, posinf=cap, neginf=0.0)
        D_pred = np.nan_to_num(D_pred, nan=0.0, posinf=cap, neginf=0.0)
        O_pred = np.clip(O_pred, 1e-9, cap)
        D_pred = np.clip(D_pred, 1e-9, cap)
        dist_matrix = np.nan_to_num(dist_matrix, nan=5.5, posinf=5.5, neginf=1e-3)

        total_O = np.sum(O_pred)
        total_D = np.sum(D_pred)
        if (not np.isfinite(total_O)) or (not np.isfinite(total_D)) or total_D <= 0:
            return np.zeros((num_nodes, num_nodes), dtype=np.float64)
        D_pred = D_pred * (total_O / total_D)
        D_pred = np.nan_to_num(D_pred, nan=1e-9, posinf=cap, neginf=1e-9)
        D_pred = np.clip(D_pred, 1e-9, cap)
        
        dist_safe = np.maximum(dist_matrix, 1e-3)
        f_d = np.nan_to_num(dist_safe ** (-self.beta), nan=0.0, posinf=1e12, neginf=0.0)

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
            T_check = np.nan_to_num(T_check, nan=0.0, posinf=cap, neginf=0.0)
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
        T_ext = np.nan_to_num(T_ext, nan=0.0, posinf=cap, neginf=0.0)

        row_error = np.max(np.abs(T_ext.sum(axis=1) - O_pred))
        col_error = np.max(np.abs(T_ext.sum(axis=0) - D_pred))

        print(
            f"IPF row error: {row_error:.6f}, "
            f"col error: {col_error:.6f}"
        )

        return T_ext

    def fit_predict(self, X_static_train, O_train, D_train, dist_matrix, useLog):
        self.fit_O_D(X_static_train, O_train, D_train, useLog)
        O_pred, D_pred = self.predict_O_D(X_static_train, useLog)
        
        dist_no_diag = dist_matrix.copy()
        np.fill_diagonal(dist_no_diag, np.inf)
        intrazonal_dist = dist_no_diag.min(axis=1) / 2.0
        
        dist_matrix_modified = dist_matrix.copy()
        np.fill_diagonal(dist_matrix_modified, intrazonal_dist)
        
        T_pred = self.apply_ipf(O_pred, D_pred, dist_matrix_modified)
        return T_pred
