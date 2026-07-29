import torch
import torch.nn as nn

class ODGCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear_in = nn.Linear(in_features, out_features)
        self.linear_out = nn.Linear(in_features, out_features)
        
    def forward(self, x_od, feat_emb, observed_mask):
        # x_od: (B, N, N) - od matrix
        # feat_emb: (B, N, D) - 각 노드의 임베딩 벡터
        # observed_mask: (B, N, N) - boolean mask, 관측 가능한 노드 쌍만 True
        
        A = x_od.clone()
        # self-loop 제거 및 관측되지 않은 노드 0으로
        A.diagonal(dim1=-2, dim2=-1).zero_() 
        A[~observed_mask] = 0.0 
        
        # Outgoing Normalize adjacency
        deg_out = A.sum(dim=-1, keepdim=True) + 1e-5
        A_norm_out = A / deg_out
        
        # Incoming Normalize adjacency (transpose)
        A_t = A.transpose(1, 2)
        deg_in = A_t.sum(dim=-1, keepdim=True) + 1e-5
        A_norm_in = A_t / deg_in
        
        # Message passing
        msg_out = torch.bmm(A_norm_out, feat_emb)
        msg_in = torch.bmm(A_norm_in, feat_emb)
        
        return self.linear_out(msg_out) + self.linear_in(msg_in)

class ODMAE(nn.Module):
    def __init__(self, num_features, d_model=128, nhead=8, num_layers=4,
                 od_embed_layers=2, use_distance_friction=True, use_self_loop_predictor=True, use_mask_channel=False):
        super().__init__()
        self.use_distance_friction = use_distance_friction
        self.od_embed_layers = od_embed_layers
        self.use_self_loop_predictor = use_self_loop_predictor
        self.use_mask_channel = use_mask_channel

        # X_static embeding: (B, N, F) -> (B, N, D) - leanable
        # OD feature embedding: (B, N, 2N or 3N) -> (B, N, D) - leanable
        self.feature_embed = nn.Sequential(
            nn.Linear(num_features, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        self.od_gcn = ODGCNLayer(d_model, d_model)
        od_in_dim = 3 if self.use_mask_channel else 2
        
        if self.od_embed_layers == 3:
            self.od_embed = nn.Sequential(
                nn.Linear(od_in_dim, d_model * 2),
                nn.GELU(),
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model)
            )
        elif self.od_embed_layers == 2:
            self.od_embed = nn.Sequential(
                nn.Linear(od_in_dim, d_model * 2),
                nn.GELU(),
                nn.Linear(d_model * 2, d_model)
            )
        else:
            self.od_embed = nn.Linear(od_in_dim, d_model)
        
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
        
        # 디코더 최종 출력에 직접 더해지는 거리 편향 (Friction)
        self.distance_decode_bias = nn.Embedding(50, 1)
        
        # Mask Token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # decoder: (B, N, D) -> (B, N, 2N)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model * 2)
        )

    def forward(self, x_static, x_od_masked, x_dist, mask, active_node_mask=None):
        """
        x_static: (B, N, F)
        x_od_masked: (B, N, N)
        x_dist: (B, N, N) distance matrix (log-scaled)
        mask: (B, N) boolean mask where True means masked (predict this)
        active_node_mask: (B, N) boolean mask where False means the node is deactivated (merged/deleted)
        """
        B, N, _ = x_static.shape
            
        if active_node_mask is None:
            active_node_mask = torch.ones(B, N, dtype=torch.bool, device=x_static.device)
            
        observed_1d = (~mask) & active_node_mask
        observed_mask_2d = observed_1d.unsqueeze(1) & observed_1d.unsqueeze(2)

        # 1) 제외해야 할 정보 차단 (self-loop 제외)
        x_od_no_diag = x_od_masked.clone()
        x_od_no_diag.diagonal(dim1=-2, dim2=-1).zero_()
        
        # 2) Masked Mean 연산 (관측된 이웃 개수로만 나누기)
        observed_col_mask = observed_1d.unsqueeze(1).expand_as(x_od_no_diag) # (B, N, N)
        row_sum = x_od_no_diag.sum(dim=-1, keepdim=True)
        row_count = observed_col_mask.float().sum(dim=-1, keepdim=True).clamp(min=1)
        row_feat = row_sum / row_count
        
        observed_row_mask = observed_1d.unsqueeze(2).expand_as(x_od_no_diag)
        col_sum = x_od_no_diag.sum(dim=-2, keepdim=True).transpose(1, 2)
        col_count = observed_row_mask.float().sum(dim=-2, keepdim=True).transpose(1, 2).clamp(min=1)
        col_feat = col_sum / col_count
        
        if self.use_mask_channel:
            mask_feat = mask.float().unsqueeze(-1)
            node_od_feat = torch.cat([row_feat, col_feat, mask_feat], dim=-1)  # (B, N, 3)
        else:
            node_od_feat = torch.cat([row_feat, col_feat], dim=-1)  # (B, N, 2)
        
        # feat_emb: (B, N, D), od_emb: (B, N, D) - 임베딩, gcn_emb: (B, N, D) - GCN 임베딩
        feat_emb = self.feature_embed(x_static) 
        od_emb = self.od_embed(node_od_feat)   
        gcn_emb = self.od_gcn(x_od_masked, feat_emb, observed_mask_2d)
        
        # mask_expanded = (B, N, D) 
        mask_expanded = mask.unsqueeze(-1).expand_as(od_emb)
        mask_token_expanded = self.mask_token.expand(B, N, -1)
        
        # 가려진 도시는 OD 정보만 마스크 토큰으로 치환
        od_emb_masked = torch.where(mask_expanded, mask_token_expanded, od_emb)
        
        # combined: (B, N, 2D)
        combined = torch.cat([feat_emb, od_emb_masked], dim=-1)
        
        # (B, N, 2D) -> (B, N, D)
        gate_val = self.od_gate(combined) 
        
        # (B, N, D) - OD 정보의 반영 비율을 조절
        x = feat_emb + (gate_val * od_emb_masked) + gcn_emb
        
        # Bucketize distance
        distance_bins = torch.bucketize(x_dist, self.boundaries) # (B, N, N) # type: ignore
        
        # bias: (B, N, N, nhead) - 각 distance bin에 대해 nhead 차원의 bias를 가져옴
        bias = self.distance_bias(distance_bins) 
        
        # bias: (B, N, N, nhead) -> (B * nhead, N, N)
        bias = bias.permute(0, 3, 1, 2).reshape(B * self.nhead, N, N)
        
        # Transformer (bias 적용)
        x = self.transformer(x, mask=bias) # (B, N, D)
        
        node_repr = self.decoder(x)  # (B, N, 2D) — 최종 노드 표현
        out_repr, in_repr = node_repr.chunk(2, dim=-1)  # (B, N, D), (B, N, D)

        # 두 관점(outgoing 기준 / incoming 기준)에서 각각 예측 후 대칭 평균
        pred_from_out_view = torch.bmm(out_repr, in_repr.transpose(1, 2))                  # (B, N, N)
        pred_from_in_view = torch.bmm(in_repr, out_repr.transpose(1, 2)).transpose(1, 2)   # (B, N, N)
        pred_od = (pred_from_out_view + pred_from_in_view) / 2.0

        if self.use_distance_friction:
            friction = self.distance_friction(distance_bins).squeeze(-1)  # (B, N, N)
            pred_od = pred_od + friction

        # 디코더 출력에 직접 더해지는 거리 편향 (distance_decode_bias 활용)
        decode_bias = self.distance_decode_bias(distance_bins).squeeze(-1)  # (B, N, N)
        pred_od = pred_od + decode_bias
        
        if self.use_self_loop_predictor:
            self_loop_pred = self.self_loop_predictor(feat_emb).squeeze(-1) # (B, N, D) -> (B, N)
        else:
            self_loop_pred = 0

        # Batch 차원과 Node 차원을 위한 인덱스 생성
        b_idx = torch.arange(B).unsqueeze(-1) # (B, 1)
        n_idx = torch.arange(N).unsqueeze(0)  # (1, N)
        
        pred_od[b_idx, n_idx, n_idx] += self_loop_pred
        
        return pred_od
