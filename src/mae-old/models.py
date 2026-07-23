import torch
import torch.nn as nn

class ODGCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear_in = nn.Linear(in_features, out_features)
        self.linear_out = nn.Linear(in_features, out_features)
        
    def forward(self, x_od, feat_emb, observed_mask):
        # x_od: (B, N, N)
        # feat_emb: (B, N, D)
        # observed_mask: (B, N, N) boolean mask
        
        # OD 매트릭스 자체를 adjacency로 사용
        # 단, mask된 노드끼리의 가짜(0) 연결성을 차단하기 위해 observed_mask 적용
        A = x_od.clone()
        A.diagonal(dim1=-2, dim2=-1).zero_() # Self-loop 방지
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

class SpatialODMAE(nn.Module):
    def __init__(self, num_nodes, num_features=14, d_model=128, nhead=8, num_layers=4, use_self_loop_predictor=True):
        super().__init__()
        self.use_self_loop_predictor = use_self_loop_predictor

        # X_static embeding: (B, N, F) -> (B, N, D) - leanable
        # OD feature embedding: (B, N, 2N or 3N) -> (B, N, D) - leanable
        self.feature_embed = nn.Sequential(
            nn.Linear(num_features, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        self.od_gcn = ODGCNLayer(d_model, d_model)
        
        # Node-invariant OD Embedder (GCN 스타일 엣지 임베딩)
        # 자기 자신으로 들어오고 나가는 통행량을 집계해서 임베딩
        od_in_dim = 3 
        
        self.od_embed = nn.Sequential(
            nn.Linear(od_in_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
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
        # 0부터 5.5 구간을 49개로 나눔
        self.register_buffer('boundaries', torch.linspace(0, 5.5, 49))
        
        # Mask Token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Bilinear Decoder for node-invariant output
        # out: (B, N, D) @ (B, D, N) -> (B, N, N)
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)

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
        
        mask_feat = mask.float().unsqueeze(-1)
        node_od_feat = torch.cat([row_feat, col_feat, mask_feat], dim=-1) # (B, N, 3)

        # feat_emb: (B, N, D), od_emb: (B, N, D) - 임베딩
        feat_emb = self.feature_embed(x_static) 
        od_emb = self.od_embed(node_od_feat)   
        
        # ODGCNLayer (그래프 구조 기반 메시지 패싱) 적용
        gcn_emb = self.od_gcn(x_od_masked, feat_emb, observed_mask_2d)
        
        # 기존 OD 피처 임베딩에 GCN 구조적 임베딩 더하기
        od_emb = od_emb + gcn_emb
        
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
        x = feat_emb + (gate_val * od_emb_masked)
        
        # Bucketize distance
        distance_bins = torch.bucketize(x_dist, self.boundaries) # (B, N, N) # type: ignore
        
        # bias: (B, N, N, nhead) - 각 distance bin에 대해 nhead 차원의 bias를 가져옴
        bias = self.distance_bias(distance_bins) 
        
        # bias: (B, N, N, nhead) -> (B * nhead, N, N)
        bias = bias.permute(0, 3, 1, 2).reshape(B * self.nhead, N, N)
        
        # Transformer (bias 적용 및 padding mask 적용)
        padding_mask = ~active_node_mask # True인 위치가 Attention에서 무시됨
        x = self.transformer(x, mask=bias, src_key_padding_mask=padding_mask) # (B, N, D)
        
        # Bilinear Decode
        queries = self.query_proj(x)
        keys = self.key_proj(x)
        
        # pred_od: (B, N, N)
        pred_od = torch.bmm(queries, keys.transpose(1, 2)) / (self.query_proj.out_features ** 0.5)
        
        if self.use_self_loop_predictor:
            self_loop_pred = self.self_loop_predictor(feat_emb).squeeze(-1) # (B, N, D) -> (B, N)
            self_loop_pred = self_loop_pred * active_node_mask.float() # 비활성 노드는 예측값을 0으로
        else:
            self_loop_pred = 0

        # Batch 차원과 Node 차원을 위한 인덱스 생성
        b_idx = torch.arange(B).unsqueeze(-1) # (B, 1)
        n_idx = torch.arange(N).unsqueeze(0)  # (1, N)
        
        pred_od[b_idx, n_idx, n_idx] += self_loop_pred
        
        import torch.nn.functional as F
        pred_od_scale = F.softplus(pred_od)
        
        return pred_od_scale, pred_od
