import torch
import torch.nn as nn

class SpatialODMAE(nn.Module):
    def __init__(self, num_features=16, d_model=128, nhead=8, num_layers=4,
                 use_distance_friction=True, use_self_loop_predictor=True):
        """
        Inductive Architecture for SpatialODMAE.
        Does not depend on the number of nodes (N).
        """
        super().__init__()
        self.use_distance_friction = use_distance_friction
        self.use_self_loop_predictor = use_self_loop_predictor

        # X_static embeding: (B, N, F) -> (B, N, D)
        self.feature_embed = nn.Sequential(
            nn.Linear(num_features, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        # Structural OD feature embedding (Self-loop, Out-degree, In-degree, Mask indicator)
        od_in_dim = 4
        self.od_embed = nn.Sequential(
            nn.Linear(od_in_dim, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
        
        # OD 정보의 반영 비율을 조절하는 Learnable Gating Network
        self.od_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
        # 자기동 내부 통행량 직접 예측을 위한 작은 MLP
        if self.use_self_loop_predictor:
            self.self_loop_predictor = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 1)
            )
        
        # distance based 상대 positional bias 및 최종 Friction
        self.nhead = nhead
        self.distance_bias = nn.Embedding(50, nhead)
        self.distance_friction = nn.Embedding(50, 1)
        # 0부터 5.5 구간을 49개로 나눔
        self.register_buffer('boundaries', torch.linspace(0, 5.5, 49))
        
        # Mask Token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # --- Auxiliary Task: Static Feature Decoder ---
        self.static_decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_features)
        )
        
        # --- Inductive Task: Pairwise OD Decoder ---
        self.pairwise_decoder = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)
        )

    def forward(self, x_static, x_od_masked, x_dist, mask):
        """
        x_static: (B, N, F)
        x_od_masked: (B, N, N)
        x_dist: (B, N, N) distance matrix (log-scaled)
        mask: (B, N) boolean mask where True means masked (predict this)
        """
        B, N, _ = x_static.shape
            
        # --- 1. Extract Structural Features from x_od_masked ---
        idx = torch.arange(N, device=x_static.device)
        self_loop = x_od_masked[:, idx, idx].unsqueeze(-1) # (B, N, 1)
        out_degree = x_od_masked.sum(dim=-1).unsqueeze(-1) - self_loop # (B, N, 1)
        in_degree = x_od_masked.sum(dim=-2).unsqueeze(-1) - self_loop  # (B, N, 1)
        mask_feat = mask.float().unsqueeze(-1) # (B, N, 1)
        
        node_od_feat = torch.cat([self_loop, out_degree, in_degree, mask_feat], dim=-1) # (B, N, 4)
        
        # --- 2. Embedding & Gating ---
        feat_emb = self.feature_embed(x_static) 
        od_emb = self.od_embed(node_od_feat)   
        
        mask_expanded = mask.unsqueeze(-1).expand_as(od_emb)
        mask_token_expanded = self.mask_token.expand(B, N, -1)
        od_emb_masked = torch.where(mask_expanded, mask_token_expanded, od_emb)
        
        combined = torch.cat([feat_emb, od_emb_masked], dim=-1)
        gate_val = self.od_gate(combined) 
        x = feat_emb + (gate_val * od_emb_masked)
        
        # --- 3. Transformer with Distance Bias ---
        distance_bins = torch.bucketize(x_dist, self.boundaries) # (B, N, N)
        bias = self.distance_bias(distance_bins) 
        bias = bias.permute(0, 3, 1, 2).reshape(B * self.nhead, N, N)
        
        x = self.transformer(x, mask=bias) # (B, N, D)
        
        # --- 4. Auxiliary Task: Predict Static Features ---
        pred_static = self.static_decoder(x) # (B, N, F)
        
        # --- 5. Pairwise OD Decoding ---
        x_row = x.unsqueeze(2).expand(B, N, N, -1) # (B, N, N, D)
        x_col = x.unsqueeze(1).expand(B, N, N, -1) # (B, N, N, D)
        pair_feat = torch.cat([x_row, x_col], dim=-1) # (B, N, N, 2D)
        
        pred_od = self.pairwise_decoder(pair_feat).squeeze(-1) # (B, N, N)
        
        if self.use_distance_friction:
            friction = self.distance_friction(distance_bins).squeeze(-1) # (B, N, N)
            pred_od = pred_od + friction
        
        if self.use_self_loop_predictor:
            self_loop_pred = self.self_loop_predictor(feat_emb).squeeze(-1) # (B, N, D) -> (B, N)
        else:
            self_loop_pred = 0

        b_idx = torch.arange(B).unsqueeze(-1)
        n_idx = torch.arange(N).unsqueeze(0)
        pred_od[b_idx, n_idx, n_idx] += self_loop_pred
        
        return pred_od, pred_static
