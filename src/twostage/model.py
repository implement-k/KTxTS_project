import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import lightgbm as lgb
import joblib

class Stage1Model_LGBM:
    '''
        Stage1: train과 val/test를 구분하여 자기 동 내부 통행량과 타 지역 간 통행량을 예측
    '''
    
    def __init__(self, normal_self=None, normal_inter=None, masked_self=None, masked_inter=None):
        params = {'objective': 'regression', 'n_estimators': 300, 'num_leaves': 15, 'min_child_samples': 10, 'n_jobs': 1}
        self.normal_self = normal_self or lgb.LGBMRegressor(**params)
        self.normal_inter = normal_inter or lgb.LGBMRegressor(**params)
        self.masked_self = masked_self or lgb.LGBMRegressor(**params)
        self.masked_inter = masked_inter or lgb.LGBMRegressor(**params)

    def fit(self, X_static, y_self, y_inter, masking_indices):
        """
        X_static: (N, F)
        y_self: (N, 1)
        y_inter: (N, 1)
        masking_indices: 강제 마스킹할 컬럼 인덱스 리스트
        """
        # Normal fit
        self.normal_self.fit(X_static, y_self)
        self.normal_inter.fit(X_static, y_inter)
        
        # Masked fit (종사자수/사업체수 강제 마스킹)
        X_static_masked = X_static.copy()
        for idx in masking_indices:
            X_static_masked[:, idx] = 0.0
        X_static_masked[:, -1] = 1.0 # is_masked = 1
        
        self.masked_self.fit(X_static_masked, y_self)
        self.masked_inter.fit(X_static_masked, y_inter)

    def predict(self, X_static):
        """
        X_static: (N, F)
        """
        is_masked = (X_static[:, -1] == 1.0)
        
        threshold_log = np.log1p(1e-6) 
        
        log_self = np.zeros(len(X_static))
        log_inter = np.zeros(len(X_static))
        
        if (~is_masked).any():
            log_self[~is_masked] = self.normal_self.predict(X_static[~is_masked])
            log_inter[~is_masked] = self.normal_inter.predict(X_static[~is_masked])
            
        if is_masked.any():
            log_self[is_masked] = self.masked_self.predict(X_static[is_masked])
            log_inter[is_masked] = self.masked_inter.predict(X_static[is_masked])
            
        return np.maximum(log_self, threshold_log), np.maximum(log_inter, threshold_log)
    
    def fit_predict(self, X_static, y_self, y_inter, masking_indices, fold=None):
        import os
        self.fit(X_static, y_self, y_inter, masking_indices)
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        suffix = f'_fold_{fold}.pkl' if fold is not None else '.pkl'
        
        joblib.dump(self.normal_self, os.path.join(current_dir, f'lgbm_normal_self{suffix}'))
        joblib.dump(self.normal_inter, os.path.join(current_dir, f'lgbm_normal_inter{suffix}'))
        joblib.dump(self.masked_self, os.path.join(current_dir, f'lgbm_masked_self{suffix}'))
        joblib.dump(self.masked_inter, os.path.join(current_dir, f'lgbm_masked_inter{suffix}'))
        
        return self.predict(X_static)


# ==== Two-Stage Gravity Model ====
class Stage2Model(nn.Module):
    def __init__(self, num_features=13, hidden_dim=64, dropout_p=0.3):
        super().__init__()
        
        # Stage1: Generation Model (Pre-trained LGBM)
        
        # Stage 2: Distribution Model 
        dim_input = num_features * 2 + 1
        layers = []
        in_dim = dim_input
        out_dim = hidden_dim
        
        for _ in range(5):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(dropout_p))
            in_dim = out_dim
            
        out_dim = hidden_dim // 2
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(nn.LeakyReLU())
        layers.append(nn.Dropout(dropout_p))
        in_dim = out_dim
        
        for _ in range(3): # slightly smaller than 15 layers for faster training
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(dropout_p))
            in_dim = out_dim
            
        self.stage2_mlp = nn.Sequential(*layers)
        self.stage2_out = nn.Linear(out_dim, 1)

    def forward(self, x_static, dist_matrix, log_self, log_inter):
        """
        x_static: (B, N, F)
        dist_matrix: (B, N, N)
        log_self: (B, N) - pre-computed by LGBM
        log_inter: (B, N) - pre-computed by LGBM
        """
        B, N, F_dim = x_static.shape
        
        # === Stage 1: Generation ===
        # Handled externally. Inputs are ready to use.
        
        # === Stage 2: Distribution ===
        # O_feat: (B, N, 1, F)
        o_feat = x_static.unsqueeze(2).expand(-1, -1, N, -1)
        # D_feat: (B, 1, N, F)
        d_feat = x_static.unsqueeze(1).expand(-1, N, -1, -1)
        # Dist_feat: (B, N, N, 1)
        dist_feat = dist_matrix.unsqueeze(-1)
        
        combined = torch.cat([o_feat, d_feat, dist_feat], dim=-1)
        hidden = self.stage2_mlp(combined)
        logits = self.stage2_out(hidden).squeeze(-1) # (B, N, N)
        
        # Mask diagonal (j == i) to -inf before softmax
        diag_mask = torch.eye(N, device=logits.device, dtype=torch.bool).unsqueeze(0).expand(B, -1, -1)
        logits = logits.masked_fill(diag_mask, float('-inf'))
        
        # Calculate log_p_ij (log-softmax over destination j)
        log_p_ij = F.log_softmax(logits, dim=2) # (B, N, N)
        
        # === Combine in Log Space ===
        # log_flow_ij = log_inter + log_p_ij
        log_flow_ij = log_inter.unsqueeze(2) + log_p_ij # (B, N, N)
        
        # The true y_OD is given in log scale: log1p(flow).
        # We know log1p(flow) = log1p(exp(log_flow)) = softplus(log_flow).
        # This mathematically avoids all vanishing gradient problems!
        pred_ij = F.softplus(log_flow_ij) # (B, N, N)
        pred_ii = F.softplus(log_self)    # (B, N)
        
        # Substitute the diagonal
        out_log = pred_ij.masked_fill(diag_mask, 0.0) + torch.diag_embed(pred_ii)
        
        return out_log

