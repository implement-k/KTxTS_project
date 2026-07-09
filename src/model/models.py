import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from xgboost import XGBRegressor

from scipy.optimize import minimize

# ==== gravity model ====
class GravityModel:
    def __init__(self):
        self.b = -1.0 # Inverse power parameter
        self.c = -0.1 # Exponential parameter

    def fit(self, P, A, dist_matrix, y_matrix, train_mask):
        """
        Fit gravity parameters using scipy minimize.
        Formula: T_ij = P_i * (A_j * F_ij) / sum_k (A_k * F_ik)
        Friction: F_ij = dist_ij^b * exp(c * dist_ij)
        """
        P = np.clip(P, 1e-5, None)
        A = np.clip(A, 1e-5, None)
        dist = np.clip(dist_matrix, 1.0, None)
        
        def loss_func(params):
            b, c = params
            # F_ij = d^b * e^(c*d)
            F = (dist ** b) * np.exp(c * dist)
            
            # A_j * F_ij
            AF = A.reshape(1, -1) * F
            
            # sum_k (A_k, F_ik)
            sum_AF = np.clip(AF.sum(axis=1, keepdims=True), 1e-9, None)
            
            # T_ij
            T_pred = P.reshape(-1, 1) * (AF / sum_AF)
            
            # MSE on train_mask
            err = T_pred[train_mask] - y_matrix[train_mask]
            return np.mean(err**2)

        p0 = [self.b, self.c]
        bounds = [(-10.0, 5.0), (-1.0, 1.0)]
        
        try:
            res = minimize(loss_func, p0, bounds=bounds, method='L-BFGS-B')
            self.b, self.c = res.x
            print(f"중력 모형 하이퍼파라미터: b={self.b:.4f}, c={self.c:.4f}")
        except Exception as e:
            print("Optimization 실패:", e)

    def predict(self, P, A, dist_matrix):
        P = np.clip(P, 1e-5, None)
        A = np.clip(A, 1e-5, None)
        dist = np.clip(dist_matrix, 1.0, None)
        
        F = (dist ** self.b) * np.exp(self.c * dist)
        AF = A.reshape(1, -1) * F
        sum_AF = AF.sum(axis=1, keepdims=True)
        sum_AF = np.clip(sum_AF, 1e-9, None)
        
        T_pred = P.reshape(-1, 1) * (AF / sum_AF)
        return T_pred

# ==== XGBoost ====
class XGBGravityModel:
    def __init__(self):
        self.model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)

    def fit(self, X_tabular, y):
        self.model.fit(X_tabular, y)

    def predict(self, X_tabular):
        return self.model.predict(X_tabular)

# ==== deep grvity ====
class DeepGravity(nn.Module):
    def __init__(self, num_features=13, hidden_dim=64):
        super().__init__()
        self.o_mlp = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.d_mlp = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.dist_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.out_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x_static, dist_matrix):
        """
        x_static: (B, N, F)
        dist_matrix: (B, N, N)
        """
        B, N, F = x_static.shape
        
        # O_feat: (B, N, 1, H) -> Broadcast to (B, N, N, H)
        o_feat = self.o_mlp(x_static).unsqueeze(2).expand(-1, -1, N, -1)
        
        # D_feat: (B, 1, N, H) -> Broadcast to (B, N, N, H)
        d_feat = self.d_mlp(x_static).unsqueeze(1).expand(-1, N, -1, -1)
        
        # Dist_feat: (B, N, N, H)
        dist_feat = self.dist_mlp(dist_matrix.unsqueeze(-1))
        
        # Concat: (B, N, N, 3*H)
        combined = torch.cat([o_feat, d_feat, dist_feat], dim=-1)
        
        # Output: (B, N, N, 1) -> (B, N, N)
        out = self.out_mlp(combined).squeeze(-1)
        return out

# ==== Spatial OD-MAE(ours) 1-channel ====
class SpatialODMAE1(nn.Module):
    def __init__(self, num_nodes=1152, num_features=13, d_model=128, nhead=8, num_layers=4):
        super().__init__()
        self.num_nodes = num_nodes
        
        # stataic feature embedding
        self.feature_embed = nn.Linear(num_features, d_model)
        
        # OD feature embedding (row i and col i for each node)
        self.od_embed = nn.Linear(num_nodes * 2, d_model)
        
        # Distance-based Relative Positional Bias
        self.nhead = nhead
        self.distance_bias = nn.Embedding(50, nhead)
        # log1p boundaries from 0 to 5.5 (covers ~243km)
        self.register_buffer('boundaries', torch.linspace(0, 5.5, 49))
        
        # Mask Token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # decoder to reconstruct OD matrix
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_nodes * 2)
        )

    def forward(self, x_static, x_od_masked, x_dist, mask):
        """
        x_static: (B, N, F)
        x_od_masked: (B, N, N)
        x_dist: (B, N, N) distance matrix (log-scaled)
        mask: (B, N) boolean mask where True means masked (predict this)
        """
        B, N, _ = x_static.shape
        
        # Extract row i and col i for each node i to form node OD features
        # x_od_masked shape: (B, N, N)
        # Row i: x_od_masked[:, i, :] -> (B, N)
        # Col i: x_od_masked[:, :, i] -> (B, N)
        # Concat -> (B, N, 2N)
        
        # Note: Since x_od_masked is symmetric conceptually in how we gather, we just transpose
        row_feat = x_od_masked
        col_feat = x_od_masked.transpose(1, 2)
        node_od_feat = torch.cat([row_feat, col_feat], dim=-1) # (B, N, 2N)
        
        # Embeddings
        feat_emb = self.feature_embed(x_static) # (B, N, D)
        
        # OD 임베딩에만 마스킹 적용
        od_emb = self.od_embed(node_od_feat)    # (B, N, D)
        mask_expanded = mask.unsqueeze(-1).expand_as(od_emb)
        mask_token_expanded = self.mask_token.expand(B, N, -1)
        
        # 가려진 도시는 OD 정보만 마스크 토큰으로 치환
        od_emb_masked = torch.where(mask_expanded, mask_token_expanded, od_emb)
        
        # 최종 결합 (인프라 정보 + 마스킹된 OD 정보)
        x = feat_emb + od_emb_masked
        
        # 1. Bucketize distance
        distance_bins = torch.bucketize(x_dist, self.boundaries) # (B, N, N)
        
        # 2. Get learned bias for each distance bin
        bias = self.distance_bias(distance_bins) # (B, N, N, nhead)
        
        # 3. Reshape bias for PyTorch Transformer src_mask
        # PyTorch expects (B * nhead, N, N)
        bias = bias.permute(0, 3, 1, 2) # (B, nhead, N, N)
        bias = bias.reshape(B * self.nhead, N, N) # (B * nhead, N, N)
        
        # Transformer (apply bias as src_mask)
        x = self.transformer(x, mask=bias) # (B, N, D)
        
        # Decode
        out = self.decoder(x) # (B, N, 2N)
        
        # Reconstruct the N x N matrix
        # Average the row predictions and col predictions for symmetric agreement
        # out[..., :N] is the predicted outgoing from node i to all j
        # out[..., N:] is the predicted incoming to node i from all j
        pred_outgoing = out[..., :N] # (B, N, N) -> pred_outgoing[b, i, j] is trip from i to j
        pred_incoming = out[..., N:].transpose(1, 2) # (B, N, N) -> pred_incoming[b, i, j] is trip from i to j
        
        # Average
        pred_od = (pred_outgoing + pred_incoming) / 2.0
        
        return pred_od

# ==== Spatial OD-MAE(ours) 5-channel ====
class SpatialODMAE5(nn.Module):
    def __init__(self, num_nodes=1152, num_features=13, d_model=128, nhead=8, num_layers=4):
        super().__init__()
        self.num_nodes = num_nodes
        
        # stataic feature embedding
        self.feature_embed = nn.Linear(num_features, d_model)
        
        # OD feature embedding (row i and col i for each node, 5 channels)
        self.od_embed = nn.Linear(num_nodes * 2 * 5, d_model)
        
        # Distance-based Relative Positional Bias
        self.nhead = nhead
        self.distance_bias = nn.Embedding(50, nhead)
        # log1p boundaries from 0 to 5.5 (covers ~243km)
        self.register_buffer('boundaries', torch.linspace(0, 5.5, 49))
        
        # Mask Token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # decoder to reconstruct OD matrix
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_nodes * 2 * 5)
        )

    def forward(self, x_static, x_od_masked, x_dist, mask):
        """
        x_static: (B, N, F)
        x_od_masked: (B, N, N, 5)
        x_dist: (B, N, N) distance matrix (log-scaled)
        mask: (B, N) boolean mask where True means masked
        """
        B, N, _ = x_static.shape
        
        # Flatten the OD features for embedding
        # row_feat: (B, N, N, 5) -> reshape to (B, N, N*5)
        row_feat = x_od_masked.reshape(B, N, N * 5)
        # col_feat: (B, N, N, 5) -> transpose to (B, N, N, 5) then reshape
        col_feat = x_od_masked.transpose(1, 2).reshape(B, N, N * 5)
        
        node_od_feat = torch.cat([row_feat, col_feat], dim=-1) # (B, N, 2N*5)
        
        # Embeddings
        feat_emb = self.feature_embed(x_static) # (B, N, D)
        od_emb = self.od_embed(node_od_feat)    # (B, N, D)
        
        mask_expanded = mask.unsqueeze(-1).expand_as(od_emb)
        mask_token_expanded = self.mask_token.expand(B, N, -1)
        
        od_emb_masked = torch.where(mask_expanded, mask_token_expanded, od_emb)
        
        x = feat_emb + od_emb_masked
        
        distance_bins = torch.bucketize(x_dist, self.boundaries) # (B, N, N)
        bias = self.distance_bias(distance_bins) # (B, N, N, nhead)
        bias = bias.permute(0, 3, 1, 2).reshape(B * self.nhead, N, N) # (B*nhead, N, N)
        
        x = self.transformer(x, mask=bias) # (B, N, D)
        
        out = self.decoder(x) # (B, N, 2N*5)
        
        # Reshape to (B, N, 2, N, 5)
        out = out.view(B, N, 2, N, 5)
        
        pred_outgoing = out[:, :, 0, :, :] # (B, N, N, 5)
        pred_incoming = out[:, :, 1, :, :].transpose(1, 2) # (B, N, N, 5)
        
        pred_od = (pred_outgoing + pred_incoming) / 2.0
        
        return pred_od
