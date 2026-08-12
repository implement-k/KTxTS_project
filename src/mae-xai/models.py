import torch
import torch.nn as nn

class ODGCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear_in = nn.Linear(in_features, out_features)
        self.linear_out = nn.Linear(in_features, out_features)
        
    def forward(self, A_spatial, feat_emb, observed_mask):
        # A_spatial: (B, N, N) - geographical adjacency matrix
        # feat_emb: (B, N, D) - 각 노드의 임베딩 벡터
        # observed_mask: (B, N, N) - boolean mask, 관측 가능한 노드 쌍만 True
        
        A = A_spatial.clone()
        # self-loop 제거 및 관측되지 않은 노드 0으로
        A.diagonal(dim1=-2, dim2=-1).zero_() 
        A[~observed_mask] = 0.0 
        
        # Outgoing Normalize adjacency
        deg_out = A.sum(dim=-1, keepdim=True)
        has_neighbor_out = deg_out > 1e-3
        A_norm_out = A / deg_out.clamp(min=1e-3)
        A_norm_out = A_norm_out * has_neighbor_out.float()
        
        # Incoming Normalize adjacency (transpose)
        A_t = A.transpose(1, 2)
        deg_in = A_t.sum(dim=-1, keepdim=True)
        has_neighbor_in = deg_in > 1e-3
        A_norm_in = A_t / deg_in.clamp(min=1e-3)
        A_norm_in = A_norm_in * has_neighbor_in.float()
        
        # Message passing
        msg_out = torch.bmm(A_norm_out, feat_emb)
        msg_in = torch.bmm(A_norm_in, feat_emb)
        
        return self.linear_out(msg_out) + self.linear_in(msg_in)

class ODCrossAttention(nn.Module):
    """
        기본 attention
        이 노드의 row(각 destination과의 flow)를, attention으로 D차원 하나에 압축.
        average pooling과 다르게, 어느 destination이 중요한지를 학습해서 가중치를 정함.
        weight는 D에만 의존, N과 무관
    """
    def __init__(self, d_model):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model))  # 학습 가능한 고정 쿼리
        self.key_proj = nn.Linear(1, d_model)   # 각 flow 값(스칼라)을 key로 projection
        self.value_proj = nn.Linear(1, d_model) # 각 flow 값을 value로 projection
        self.scale = d_model ** -0.5

    def forward(self, row_flows, observed_mask):
        '''
            row_flows: (B, N, N) - 각 노드의 outgoing 통행량(masking 되어있는 matrix)
            observed_mask: (B, N, N) - 관측 가능한 노드 쌍만 True
        '''
        
        Q = self.query.squeeze() # (D)
        W_key = self.key_proj.weight.squeeze() # (D)
        b_key = self.key_proj.bias # (D)
        
        # flow 값 기반의 스코어 연산 (N, N의 각 flow마다 독립적인 projection)
        score_weight = (Q * W_key).sum() * self.scale
        score_bias = (Q * b_key).sum() * self.scale
        
        scores = row_flows * score_weight + score_bias
        
        # 마스킹 처리: 관측 불가 노드와의 attention은 -inf
        scores = scores.masked_fill(~observed_mask, float('-inf'))
        
        # Softmax를 통해 (B, N, N) 의 attention map 생성
        attn = torch.softmax(scores, dim=-1)                # (B, N, N)
        attn = torch.nan_to_num(attn, nan=0.0)              # Prevent NaN when all scores are -inf
        
        # value 역시 flow 값에서 유도 
        # 풀어서 쓰면:
        # Values = row_flows * W_val + b_val
        # output = attn @ Values
        #        = sum_j (attn_ij * (row_flows_ij * W_val + b_val))
        #        = (sum_j attn_ij * row_flows_ij) * W_val + (sum_j attn_ij) * b_val
        W_val = self.value_proj.weight.squeeze()
        b_val = self.value_proj.bias
        
        # weighted_flows: (B, N, 1) - 각 row마다 attn * flow의 합
        weighted_flows = (attn * row_flows).sum(dim=2, keepdim=True)
        
        # sum_attn: (B, N, 1) - 각 row의 attention 합 (보통 1이지만, 모두 mask된 row는 0)
        sum_attn = attn.sum(dim=2, keepdim=True)
        
        # 최종 pooling 값: (B, N, D)
        pooled = weighted_flows * W_val + sum_attn * b_val
        
        return pooled

class GravityBase(nn.Module):
    def __init__(self, static_dim):
        super().__init__()
        self.O_weight = nn.Parameter(torch.ones(static_dim)) 
        self.D_weight = nn.Parameter(torch.ones(static_dim))
        
        self.gamma = nn.Parameter(torch.tensor(1.0)) 
        self.log_k = nn.Parameter(torch.tensor(0.0)) 

    def forward(self, x_static, x_dist, O_importance_adj=None, D_importance_adj=None):
        Ow = self.O_weight if O_importance_adj is None else self.O_weight * O_importance_adj
        Dw = self.D_weight if D_importance_adj is None else self.D_weight * D_importance_adj
        
        O_i = torch.sum(x_static * Ow, dim=-1, keepdim=True) # (B, N, 1)
        D_j = torch.sum(x_static * Dw, dim=-1, keepdim=True) # (B, N, 1)
        
        gravity_logit = O_i + D_j.transpose(1, 2) - self.gamma * x_dist + self.log_k
        return gravity_logit

class ODMAE(nn.Module):
    def __init__(self, num_features, d_model=128, nhead=8, num_layers=4):
        super().__init__()

        # X_static embeding: (B, N, F) -> (B, N, D) - leanable
        # OD feature embedding: (B, N, 2N or 3N) -> (B, N, D) - leanable
        self.feature_embed = nn.Sequential(
            nn.Linear(num_features, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        # Hybrid Components
        self.gravity = GravityBase(num_features)
        self.residual_gate = nn.Parameter(torch.tensor(0.01))
        
        # row_attn_pool + col_attn_pool 출력 D로 변환: (B, N, 2D) -> (B, N, D)
        self.od_combine = nn.Linear(d_model * 2, d_model)
        
        self.od_gcn = ODGCNLayer(d_model, d_model)
        self.od_scale_gcn = ODGCNLayer(2, d_model) 
        
        # === OD 관계 반영 === 
        self.row_attn_pool = ODCrossAttention(d_model)
        self.col_attn_pool = ODCrossAttention(d_model)
        ########################################################
        
        # OD 정보의 반영 비율을 조절하는 Learnable Gating Network
        self.od_gate = nn.Sequential(
            nn.Linear(d_model * 2 + 1, d_model),
            nn.Sigmoid()
        )
        
        # distance based 상대 positional bias 및 최종 Friction
        self.nhead = nhead
        self.distance_bias = nn.Embedding(50, nhead)
        # 0부터 5.5 구간을 49개로 나눔
        self.register_buffer('boundaries', torch.linspace(0, 5.5, 49))
        
        # 디코더 최종 출력에 직접 더해지는 거리 편향 (Friction)
        self.distance_decode_bias = nn.Embedding(50, 1)
        
        # Mask Token (비율에 따른 동적 생성)
        self.mask_token_low = nn.Parameter(torch.zeros(1, 1, d_model))
        self.mask_token_high = nn.Parameter(torch.zeros(1, 1, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # decoder: (B, N, D) -> (B, N, 2N)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model * 2)
        )

    def forward(self, x_static, x_od_masked, x_dist, A_spatial, mask, active_node_mask=None,
                O_importance_adj=None, D_importance_adj=None):
        """
        x_static: (B, N, F) - mask 노드에 대해서는 (사업체 수, 종사자 수, 밀도)등은 0으로 대체된 X_static
        x_od_masked: (B, N, N)
        x_dist: (B, N, N) distance matrix (log-scaled)
        A_spatial: (B, N, N) geographical adjacency matrix
        mask: (B, N) boolean mask where True means masked (predict this)
        active_node_mask: (B, N) boolean mask where False means the node is deactivated (merged/deleted)
        """
        # === 1. 사전 작업 ===
        B, N, _ = x_static.shape
            
        if active_node_mask is None:
            print("W: [model.forward] active_node_mask is None, assuming all nodes are active.")
            active_node_mask = torch.ones(B, N, dtype=torch.bool, device=x_static.device)
            
        # observed_1d: (B, N) - 활성화 + masked 안된 노드만 관측 가능
        # observed_mask_2d: (B, N, N) - 관측 가능한 노드 쌍만 True
        observed_1d = (~mask) & active_node_mask
        observed_mask_2d = observed_1d.unsqueeze(1) & observed_1d.unsqueeze(2)
        
        active_mask_2d = active_node_mask.unsqueeze(1) & active_node_mask.unsqueeze(2)  # (B, N, N)

        # 제외해야 할 정보 차단 (self-loop 제외)
        x_od_no_diag = x_od_masked.clone()
        x_od_no_diag.diagonal(dim1=-2, dim2=-1).zero_()
        ########################################################################
        
        # === 2. OD feature embedding(주변과 OD 관계가 어떻게 되어있지?) ===
        # row_repr: (B, N, D) - 각 노드의 outgoing 통행량을 attention으로 요약
        # col_repr: (B, N, D) - 각 노드의 incoming 통행량을 attention으로 요약
        row_repr = self.row_attn_pool(x_od_no_diag, observed_mask_2d)         
        col_repr = self.col_attn_pool(x_od_no_diag.transpose(1, 2), observed_mask_2d.transpose(1, 2))             

        # od_emb: (B, N, 2D) - 임베딩
        od_emb = self.od_combine(torch.cat([row_repr, col_repr], dim=-1))
        ########################################################################
        
        # === 3. static feature embedding(이 노드는 어떤 특성을 가지고 있지?) ===
        # feat_emb: (B, N, D)
        feat_emb = self.feature_embed(x_static)
        ########################################################################
        
        # === 4. GCN embedding(이 노드는 주변 노드들과 어떤 관계가 있지?) ===
        # 4.1. 총 유출량 평균
        row_sum = x_od_no_diag.sum(dim=-1, keepdim=True)
        row_count = observed_mask_2d.float().sum(dim=-1, keepdim=True).clamp(min=1)
        row_scale = row_sum / row_count  # (B, N, 1)

        # 4.2. 총 유입량 평균
        col_sum = x_od_no_diag.sum(dim=-2, keepdim=True).transpose(1, 2)
        col_count = observed_mask_2d.float().sum(dim=-2, keepdim=True).transpose(1, 2).clamp(min=1)
        col_scale = col_sum / col_count  # (B, N, 1)

        # 4.3. od_scale: (B, N, 2) - 각 노드의 outgoing/incoming 평균 통행량
        od_scale = torch.cat([row_scale, col_scale], dim=-1)  
        od_scale = od_scale * (~mask).unsqueeze(-1).float()  # mask=True 노드는 0(관측 안 됨을 명시)

        # inferred_od_scale: (B, N, D) - 이웃의 평균 통행량을 GCN으로 반영
        # gcn_emb: (B, N, D) - 이웃의 static feature 정보를 GCN으로 반영
        inferred_od_scale = self.od_scale_gcn(A_spatial, od_scale, active_mask_2d)
        inferred_od_scale = torch.zeros_like(inferred_od_scale)
        gcn_emb = self.od_gcn(A_spatial, feat_emb, active_mask_2d)  
        ########################################################################
        
        # === 5. OD 정보와 static feature를 합치고, mask 여부를 반영한 gating ===
        # mask 여부 (B, N) -> (B, N, D)로 확장, od_emb 대체용
        mask_expanded = mask.unsqueeze(-1).expand_as(od_emb)

        # 마스크 비율 기반 동적 토큰
        mask_ratio = mask.float().mean(dim=1, keepdim=True).unsqueeze(-1)
        mask_token = self.mask_token_low * (1.0 - mask_ratio) + self.mask_token_high * mask_ratio

        # 가려진 노드는 OD 정보를 mask_token으로 치환 
        inpainted_value = mask_token.expand(B, N, -1) + inferred_od_scale
        od_emb_masked = torch.where(mask_expanded, inpainted_value, od_emb)

        # gate에 mask 여부를 명시적으로 추가
        mask_feat_for_gate = mask.float().unsqueeze(-1)  # (B, N, 1)
        combined = torch.cat([feat_emb, od_emb_masked, mask_feat_for_gate], dim=-1)  # (B, N, 2D+1)
        gate_val = self.od_gate(combined)

        # (B, N, D) - OD 정보의 반영 비율을 조절
        x = feat_emb + (gate_val * od_emb_masked) + gcn_emb
        ########################################################################
        
        # === 6. distance 기반 bias 적용 Transformer ===
        # Bucketize distance
        distance_bins = torch.bucketize(x_dist, self.boundaries) # (B, N, N) # type: ignore
        
        # bias: (B, N, N, nhead) - 각 distance bin에 대해 nhead 차원의 bias를 가져옴
        bias = self.distance_bias(distance_bins) 
        
        # bias: (B, N, N, nhead) -> (B * nhead, N, N)
        bias = bias.permute(0, 3, 1, 2).reshape(B * self.nhead, N, N)
        
        x = self.transformer(x, mask=bias) # (B, N, D)
        
        node_repr = self.decoder(x)  # (B, N, 2D) — 최종 노드 표현
        out_repr, in_repr = node_repr.chunk(2, dim=-1)  # (B, N, D), (B, N, D)

        # 두 관점(outgoing 기준 / incoming 기준)에서 각각 예측 후 대칭 평균
        pred_from_out_view = torch.bmm(out_repr, in_repr.transpose(1, 2))                  # (B, N, N)
        pred_from_in_view = torch.bmm(in_repr, out_repr.transpose(1, 2)).transpose(1, 2)   # (B, N, N)
        pred_od = (pred_from_out_view + pred_from_in_view) / 2.0

        # 디코더 출력에 직접 더해지는 거리 편향 (distance_decode_bias 활용)
        decode_bias = self.distance_decode_bias(distance_bins).squeeze(-1)  # (B, N, N)
        pred_od = pred_od + decode_bias
        
        # Hybrid Combination
        gravity_logit = self.gravity(x_static, x_dist, O_importance_adj, D_importance_adj)
        final_logit = gravity_logit + self.residual_gate * pred_od
        
        return final_logit
