import torch
import torch.nn as nn

class ODGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.act = nn.GELU()

    def forward(self, A, H):
        """
        A: (B, N, N) Weighted adjacency matrix (e.g. OD flows)
        H: (B, N, in_dim) Node features
        """
        # 정규화: out-degree(row sum) 기준으로 메시지 스케일을 맞춥니다.
        # +1e-5는 분모가 0이 되는 것을 방지
        deg = A.sum(dim=-1, keepdim=True) + 1e-5
        A_norm = A / deg
        
        # 메시지 패싱: (B, N, N) @ (B, N, in_dim) -> (B, N, in_dim)
        msg = torch.bmm(A_norm, H)
        out = self.act(self.W(msg))
        return out

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
        
        # Structural OD feature embedding (GCN)
        self.od_gcn = ODGCNLayer(d_model, d_model)
        
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
        
        # --- Inductive Task: Pairwise OD Decoder (Memory Efficient) ---
        # OOM 방지를 위해 (B, N, N, 2D)를 생성하는 MLP 대신 Bilinear 내적(Q-K) 방식 사용
        self.decoder_q = nn.Linear(d_model, d_model)
        self.decoder_k = nn.Linear(d_model, d_model)

    def forward(self, x_static, x_od_masked, x_dist, mask):
        """
        x_static: (B, N, F)
        x_od_masked: (B, N, N)
        x_dist: (B, N, N) distance matrix (log-scaled)
        mask: (B, N) boolean mask where True means masked (predict this)
        """
        B, N, _ = x_static.shape
            
        # --- 1. Embedding Static Features ---
        feat_emb = self.feature_embed(x_static) # (B, N, D)
        
        # --- 2. GCN 기반 Structural Features 추출 ---
        # x_od_masked를 인접 행렬(A)로 사용하여 feat_emb를 섞음
        od_emb = self.od_gcn(x_od_masked, feat_emb) # (B, N, D)
        
        # Residual Connection
        x = feat_emb + od_emb
        
        
        # Masking: 마스킹된 노드는 정적 변수와 구조적 특성을 모두 숨기고 mask_token으로 대체
        mask_expanded = mask.unsqueeze(-1).expand_as(x)
        mask_token_expanded = self.mask_token.expand(B, N, -1)
        x = torch.where(mask_expanded, mask_token_expanded, x)
        
        # --- 3. Transformer with Distance Bias ---
        distance_bins = torch.bucketize(x_dist, self.boundaries) # (B, N, N)
        bias = self.distance_bias(distance_bins) 
        bias = bias.permute(0, 3, 1, 2).reshape(B * self.nhead, N, N)
        
        x = self.transformer(x, mask=bias) # (B, N, D)
        
        # --- 4. Auxiliary Task: Predict Static Features ---
        pred_static = self.static_decoder(x) # (B, N, F)
        
        # --- 5. Pairwise OD Decoding (Memory Efficient) ---
        q = self.decoder_q(x) # (B, N, D)
        k = self.decoder_k(x) # (B, N, D)
        
        # (B, N, D) @ (B, D, N) -> (B, N, N)
        pred_od = torch.bmm(q, k.transpose(1, 2)) / (q.size(-1) ** 0.5)
        
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
