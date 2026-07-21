import torch
import torch.nn as nn

class ODGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W_out = nn.Linear(in_dim, out_dim // 2, bias=False)
        self.W_in = nn.Linear(in_dim, out_dim // 2, bias=False)
        self.act = nn.GELU()

    def forward(self, A, H, observed_mask):
        """
        A: (B, N, N) Weighted adjacency matrix (e.g. OD flows)
        H: (B, N, in_dim) Node features
        observed_mask: (B, N, N) Boolean or float mask where 1/True means observed, 0/False means masked
        """
        # 1. Outgoing Message Passing (내가 어디로 나가는가)
        A_out = A * observed_mask.float()
        deg_out = A_out.sum(dim=-1, keepdim=True) + 1e-5
        A_out_norm = A_out / deg_out
        msg_out = torch.bmm(A_out_norm, H)
        out_emb = self.act(self.W_out(msg_out))
        
        # 2. Incoming Message Passing (나에게 어디서 들어오는가)
        # Transpose to get incoming edges
        A_in = A.transpose(1, 2) * observed_mask.transpose(1, 2).float()
        deg_in = A_in.sum(dim=-1, keepdim=True) + 1e-5
        A_in_norm = A_in / deg_in
        msg_in = torch.bmm(A_in_norm, H)
        in_emb = self.act(self.W_in(msg_in))
        
        # Concat (B, N, out_dim // 2) + (B, N, out_dim // 2) -> (B, N, out_dim)
        return torch.cat([out_emb, in_emb], dim=-1)

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
        
        # Spatial Positional Encoding (물리적 거리를 활용한 위치 인코딩)
        self.distance_scale = nn.Parameter(torch.tensor(1.0))
        self.spe_proj = nn.Linear(d_model, d_model)
        
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
        
        # 1. Static Features Embedding & Masking
        feat_emb_raw = self.feature_embed(x_static) # (B, N, D)
        
        # 마스킹된 노드는 정적 변수를 숨기고 mask_token으로 대체
        mask_expanded = mask.unsqueeze(-1)
        mask_token_expanded = self.mask_token.expand(B, N, -1)
        feat_emb = torch.where(mask_expanded, mask_token_expanded, feat_emb_raw)
        
        # 2. Structural Features 추출
        # 관측 여부를 나타내는 2D 마스크 생성 (마스킹된 노드와 연결된 엣지는 False(0))
        # mask: (B, N) 여기서 True가 마스킹(결측)을 의미하므로, 관측된 것은 ~mask
        observed_1d = ~mask
        observed_mask_2d = observed_1d.unsqueeze(1) & observed_1d.unsqueeze(2) # (B, N, N)
        
        # 마스킹된 노드는 x_od_masked에 연결이 끊겨 있으므로 메시지를 받지 못함
        od_emb = self.od_gcn(x_od_masked, feat_emb, observed_mask_2d) 
        
        # 3. Spatial Positional Encoding (SPE)
        # 물리적 거리(x_dist)를 기반으로 주변 노드들의 feat_emb를 가중합하여 위치 정체성 부여
        # 자기 자신(대각선)의 거리가 0이 되어 가중치가 1이 되는 현상(self-loop 붕괴) 방지
        # 대각선을 0이 아닌 작은 근사값(0.5)으로 처리하여 지배적인 영향을 완화
        x_dist_eff = x_dist.clone()
        idx = torch.arange(N, device=x_dist.device)
        x_dist_eff[:, idx, idx] = 0.5
        
        A_dist = torch.exp(- x_dist_eff / (self.distance_scale ** 2 + 1e-5))
        deg_dist = A_dist.sum(dim=-1, keepdim=True) + 1e-5
        A_dist_norm = A_dist / deg_dist
        spe = self.spe_proj(torch.bmm(A_dist_norm, feat_emb))
        
        # 4. 결합
        x = feat_emb + od_emb + spe
        
        # --- 3. Transformer ---
        # 사용자가 속도 차이가 크지 않다고 판단하여 distance_bias를 부활시킴 (FlashAttention 대신 MathAttention 사용)
        distance_bins = torch.bucketize(x_dist, self.boundaries) # (B, N, N)
        bias = self.distance_bias(distance_bins) # (B, N, N, nhead)
        
        # PyTorch TransformerEncoder mask shape requirement: (B * nhead, N, N)
        B, N, _ = x_static.shape
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
            distance_bins = torch.bucketize(x_dist, self.boundaries) # (B, N, N)
            friction = self.distance_friction(distance_bins).squeeze(-1) # (B, N, N)
            pred_od = pred_od + friction
        
        if self.use_self_loop_predictor:
            self_loop_pred = self.self_loop_predictor(feat_emb).squeeze(-1) # (B, N, D) -> (B, N)
        else:
            self_loop_pred = 0

        b_idx = torch.arange(B).unsqueeze(-1)
        n_idx = torch.arange(N).unsqueeze(0)
        pred_od[b_idx, n_idx, n_idx] += self_loop_pred
        
        # 딥러닝 내적 연산 특성상 음수가 나올 수 있으나, 
        # 타겟 값인 log1p(flow)는 항상 0 이상의 양수이므로 Softplus를 통해 깔끔하게 제어
        pred_od = torch.nn.functional.softplus(pred_od)
        
        return pred_od, pred_static
