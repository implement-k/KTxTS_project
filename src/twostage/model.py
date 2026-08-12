import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import lightgbm as lgb
import joblib
import os

class Stage1Model_LGBM:
    '''
        Stage1: 모델 버전에 따라 타겟(self/inter 또는 o/d)과 모델 수가 다름.
    '''
    
    def __init__(self, use_4_lgbm=False):
        self.use_4_lgbm = use_4_lgbm
        params = {'objective': 'regression', 'n_estimators': 300, 'num_leaves': 15, 'min_child_samples': 10, 'n_jobs': 1}
        
        if self.use_4_lgbm:
            # 원래 v3 방식: Normal과 Masked를 별도로 학습
            self.normal_1 = lgb.LGBMRegressor(**params)
            self.normal_2 = lgb.LGBMRegressor(**params)
            self.masked_1 = lgb.LGBMRegressor(**params)
            self.masked_2 = lgb.LGBMRegressor(**params)
        else:
            # v2, v4, v5: Augmented Data로 하나의 통합된 모델 쌍을 학습
            self.model_1 = lgb.LGBMRegressor(**params)
            self.model_2 = lgb.LGBMRegressor(**params)

    def fit(self, X_static, y1, y2, masking_indices):
        """
        X_static: (N, F)
        y1: (N, 1) - self(v2,v3) or o(v4,v5)
        y2: (N, 1) - inter(v2,v3) or d(v4,v5)
        """
        if self.use_4_lgbm:
            self.normal_1.fit(X_static, y1)
            self.normal_2.fit(X_static, y2)
            
            X_static_masked = X_static.copy()
            for idx in masking_indices:
                X_static_masked[:, idx] = 0.0
            X_static_masked[:, -1] = 1.0 # is_masked
            
            self.masked_1.fit(X_static_masked, y1)
            self.masked_2.fit(X_static_masked, y2)
        else:
            X_train_normal = X_static.copy()
            X_train_masked = X_static.copy()
            for idx in masking_indices:
                X_train_masked[:, idx] = np.nan
                
            X_train_augmented = np.vstack([X_train_normal, X_train_masked])
            y1_augmented = np.concatenate([y1, y1])
            y2_augmented = np.concatenate([y2, y2])
            
            self.model_1.fit(X_train_augmented, y1_augmented)
            self.model_2.fit(X_train_augmented, y2_augmented)

    def predict(self, X_static):
        threshold_log = np.log1p(1e-6) 
        
        if self.use_4_lgbm:
            is_masked = (X_static[:, -1] == 1.0)
            log_1 = np.zeros(len(X_static))
            log_2 = np.zeros(len(X_static))
            
            if (~is_masked).any():
                log_1[~is_masked] = self.normal_1.predict(X_static[~is_masked])
                log_2[~is_masked] = self.normal_2.predict(X_static[~is_masked])
                
            if is_masked.any():
                log_1[is_masked] = self.masked_1.predict(X_static[is_masked])
                log_2[is_masked] = self.masked_2.predict(X_static[is_masked])
        else:
            log_1 = self.model_1.predict(X_static)
            log_2 = self.model_2.predict(X_static)
   
        return np.maximum(log_1, threshold_log), np.maximum(log_2, threshold_log)

    def fit_predict(self, X_static, y1, y2, masking_indices, fold=None):
        self.fit(X_static, y1, y2, masking_indices)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        suffix = f'_fold_{fold}.pkl' if fold is not None else '.pkl'
        
        # 가중치 저장 기능
        if self.use_4_lgbm:
            joblib.dump(self.normal_1, os.path.join(current_dir, f'lgbm_normal_1{suffix}'))
            joblib.dump(self.normal_2, os.path.join(current_dir, f'lgbm_normal_2{suffix}'))
        else:
            joblib.dump(self.model_1, os.path.join(current_dir, f'lgbm_model_1{suffix}'))
            joblib.dump(self.model_2, os.path.join(current_dir, f'lgbm_model_2{suffix}'))
            
        return self.predict(X_static)


# ==== Two-Stage Gravity Model ====
class Stage2Model(nn.Module):
    def __init__(self, num_features=13, hidden_dim=64, dropout_p=0.3, use_od=False, predict_only_masked=False, use_residual=False):
        super().__init__()
        self.use_od = use_od
        self.predict_only_masked = predict_only_masked
        self.use_residual = use_residual
        
        # Stage 2: Distribution Model 
        if not self.use_od: # originally v4/v5
            # v4/v5: [O_feat, log_O, D_feat, log_D, Dist_feat]
            dim_input = (num_features + 1) * 2 + 1
        else:
            # v2, v3: [O_feat, D_feat, Dist_feat]
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
        
        for _ in range(3): 
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(dropout_p))
            in_dim = out_dim
            
        self.stage2_mlp = nn.Sequential(*layers)
        self.stage2_out = nn.Linear(out_dim, 1)

    def forward(self, x_static, dist_matrix, log_1, log_2, mask_1d=None, true_OD=None):
        """
        log_1, log_2: Stage1 predictions (self/inter or O/D)
        """
        x_static = torch.nan_to_num(x_static, nan=0.0)
        B, N, F_dim = x_static.shape
        
        o_feat = x_static.unsqueeze(2).expand(-1, -1, N, -1)
        d_feat = x_static.unsqueeze(1).expand(-1, N, -1, -1)
        dist_feat = dist_matrix.unsqueeze(-1)
        
        if not self.use_od: # originally v4/v5
            log_O_feat = log_1.view(B, N, 1, 1).expand(-1, -1, N, 1)
            log_D_feat = log_2.view(B, 1, N, 1).expand(-1, N, -1, 1)
            combined = torch.cat([o_feat, log_O_feat, d_feat, log_D_feat, dist_feat], dim=-1)
        else:
            combined = torch.cat([o_feat, d_feat, dist_feat], dim=-1)
            
        hidden = self.stage2_mlp(combined)
        logits = self.stage2_out(hidden).squeeze(-1) # (B, N, N)
        
        diag_mask = torch.eye(N, device=logits.device, dtype=torch.bool).unsqueeze(0).expand(B, -1, -1)
        
        if not self.use_od and self.use_residual: # originally v2
            # v2 specific logic: Residual learning and manual OD replacement
            ignore_mask = (~mask_1d.unsqueeze(2)) & (~mask_1d.unsqueeze(1)) 
            full_ignore_mask = ignore_mask | diag_mask
            masked_logits = logits.masked_fill(full_ignore_mask, float('-inf'))
            
            log_p = F.log_softmax(masked_logits, dim=2) 
            O_inter_pred = torch.expm1(log_2) 
            
            known_external_mask = ignore_mask & (~diag_mask)
            known_external_flows = (true_OD * known_external_mask.float()).sum(dim=2)
            
            O_inter_residual = torch.clamp(O_inter_pred - known_external_flows, min=1e-6)
            log_O_inter_residual = torch.log1p(O_inter_residual)
            
            pred_log_flow = log_O_inter_residual.unsqueeze(2) + log_p
            pred_full_log = pred_log_flow.masked_fill(diag_mask, 0.0) + torch.diag_embed(log_1)
            pred_out_log = F.softplus(pred_full_log)
            
            true_log_OD = torch.log1p(true_OD)
            out_log = torch.where(ignore_mask, true_log_OD, pred_out_log)
            return out_log
            
        elif not self.use_od and not self.use_residual: # originally v3
            # v3 specific logic: Diagonal explicitly masked, self given directly
            logits = logits.masked_fill(diag_mask, float('-inf'))
            log_p_ij = F.log_softmax(logits, dim=2)
            log_flow_ij = log_2.unsqueeze(2) + log_p_ij
            
            pred_ij = F.softplus(log_flow_ij)
            pred_ii = F.softplus(log_1)
            out_log = pred_ij.masked_fill(diag_mask, 0.0) + torch.diag_embed(pred_ii)
            return out_log
            
        else: # originally v4, v5 (use_od is False)
            # v4, v5 specific logic: Generation / Attraction model without manual diagonal
            log_p_ij = F.log_softmax(logits, dim=2)
            log_flow_ij = log_1.unsqueeze(2) + log_p_ij
            out_log = F.softplus(log_flow_ij)
            return out_log
