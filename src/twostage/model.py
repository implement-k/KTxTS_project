import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import lightgbm as lgb
import joblib

class Stage1Model:
    '''
        Stage1: 자기 동 내부 통행량(y_self)과 타 지역 간 통행량(y_inter)을 예측하는 모델
    '''
    
    def __init__(self, model_self=None, model_inter=None):
        self.model_self = model_self or lgb.LGBMRegressor(objective='regression', n_estimators=300, num_leaves=15, min_child_samples=10)
        self.model_inter = model_inter or lgb.LGBMRegressor(objective='regression', n_estimators=300, num_leaves=15, min_child_samples=10)

    def fit(self, X_static, y_self, y_inter):
        """
        X_static: (N, F)
        y_self: (N, 1)
        y_inter: (N, 1)
        """
        self.model_self.fit(X_static, y_self)
        self.model_inter.fit(X_static, y_inter)

    def predict(self, X_static):
        """
        X_static: (N, F)
        Returns:
            log_self: (N,)
            log_inter: (N,)
        """
        threshold_log = np.log1p(1e-6) 
        log_self = np.maximum(self.model_self.predict(X_static), threshold_log)
        log_inter = np.maximum(self.model_inter.predict(X_static), threshold_log)
        
        return log_self, log_inter
    
    def fit_predict(self, X_static, y_self, y_inter):
        self.fit(X_static, y_self, y_inter)
        joblib.dump(self.model_self, 'lgbm_self.pkl')
        joblib.dump(self.model_inter, 'lgbm_inter.pkl')
        return self.predict(X_static)

# ==== Two-Stage Gravity Model ====
class Stage2Model(nn.Module):
    def __init__(self, num_features=13, hidden_dim=64, dropout_p=0.35):
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

