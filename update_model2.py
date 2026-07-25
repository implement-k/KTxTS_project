with open('src/gravity(경훈)/model.py', 'r') as f:
    text = f.read()

import re

# Since I know EXACTLY how model.py should look, I will just write a new model.py based on the existing one,
# replacing the DoublyConstrainedGravityModel class completely.

header = text.split("class DoublyConstrainedGravityModel:")[0]

new_class = """class DoublyConstrainedGravityModel:
    def __init__(self, generation_model_type='lgbm', beta=2.0, max_iter=100, tol=1e-4):
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
                n_estimators=100, learning_rate=0.05, max_depth=5, min_child_samples=5, random_state=42
            )
            self.model_D = lgb.LGBMRegressor(
                n_estimators=100, learning_rate=0.05, max_depth=5, min_child_samples=5, random_state=42
            )
        else:
            raise ValueError(f"Unknown generation_model_type: {self.generation_model_type}")

    def fit_lgbm_O_D(self, X_static, O_true, D_true):
        print(f"Training Model for Origin Generation (O_i) using {self.generation_model_type}...")
        use_log = self.generation_model_type not in ('trip_rate', 'cross_class', 'linear_regression')
        
        y_O = np.ascontiguousarray(np.log1p(O_true) if use_log else O_true, dtype=np.float64)
        y_D = np.ascontiguousarray(np.log1p(D_true) if use_log else D_true, dtype=np.float64)
        X_s = np.ascontiguousarray(X_static, dtype=np.float64)

        self.model_O.fit(X_s, y_O)

        print(f"Training Model for Destination Generation (D_j) using {self.generation_model_type}...")
        self.model_D.fit(X_s, y_D)

    def predict_O_D(self, X_static):
        use_log = self.generation_model_type not in ('trip_rate', 'cross_class', 'linear_regression')
        X_s = np.ascontiguousarray(X_static, dtype=np.float64)
        O_pred = self.model_O.predict(X_s)
        D_pred = self.model_D.predict(X_s)

        if use_log:
            O_pred = np.expm1(O_pred)
            D_pred = np.expm1(D_pred)
            
        O_pred = np.maximum(O_pred, 0)
        D_pred = np.maximum(D_pred, 0)
        return O_pred, D_pred

    def apply_ipf(self, O_pred, D_pred, dist_matrix):
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

        row_error = np.max(np.abs(T_ext.sum(axis=1) - O_pred))
        col_error = np.max(np.abs(T_ext.sum(axis=0) - D_pred))

        print(
            f"IPF row error: {row_error:.6f}, "
            f"col error: {col_error:.6f}"
        )

        return T_ext

    def fit_predict(self, X_static_train, O_train, D_train, X_static_all, dist_matrix):
        self.fit_lgbm_O_D(X_static_train, O_train, D_train)
        O_pred, D_pred = self.predict_O_D(X_static_all)
        
        dist_no_diag = dist_matrix.copy()
        np.fill_diagonal(dist_no_diag, np.inf)
        intrazonal_dist = dist_no_diag.min(axis=1) / 2.0
        
        dist_matrix_modified = dist_matrix.copy()
        np.fill_diagonal(dist_matrix_modified, intrazonal_dist)
        
        T_pred = self.apply_ipf(O_pred, D_pred, dist_matrix_modified)
        return T_pred
"""

with open('src/gravity(경훈)/model.py', 'w') as f:
    f.write(header + new_class)

