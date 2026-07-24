import torch
import torch.nn as nn
import torch.nn.functional as F

class ODGCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x_od, feat_emb, observed_mask):
        # x_od: (B, N, N)
        # feat_emb: (B, N, D)
        # observed_mask: (B, N, N) boolean mask
        
        # OD 매트릭스 자체를 adjacency로 사용
        # 단, mask된 노드끼리의 가짜(0) 연결성을 차단하기 위해 observed_mask 적용
        A = x_od.clone()
        A[~observed_mask] = 0.0 
        
        # Normalize adjacency
        deg = A.sum(dim=-1, keepdim=True) + 1e-5
        A_norm = A / deg
        
        # Message passing
        msg = torch.bmm(A_norm, feat_emb) # (B, N, N) @ (B, N, D) -> (B, N, D)
        
        return self.linear(msg)

class MetaGravityPrior(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.G_net = nn.Sequential(
            nn.Linear(d_model * 2, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        self.Alpha_net = nn.Sequential(
            nn.Linear(d_model * 2, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, feat_emb, x_dist, pop_raw):
        # feat_emb: (B, N, D)
        # x_dist: (B, N, N)
        # pop_raw: (B, N) -> log1p scale
        
        B, N, D = feat_emb.shape
        
        # Expand feat_emb to (B, N, N, D) for pairwise computation
        feat_i = feat_emb.unsqueeze(2).expand(B, N, N, D)
        feat_j = feat_emb.unsqueeze(1).expand(B, N, N, D)
        pair_feat = torch.cat([feat_i, feat_j], dim=-1) # (B, N, N, 2D)
        
        G = self.G_net(pair_feat).squeeze(-1) # (B, N, N)
        Alpha = self.Alpha_net(pair_feat).squeeze(-1) # (B, N, N)
        
        pop_i = pop_raw.unsqueeze(2).expand(B, N, N)
        pop_j = pop_raw.unsqueeze(1).expand(B, N, N)
        pop_combine = pop_i + pop_j
        
        # distance in meta-gravity is usually log-scaled or already scaled.
        # x_dist is log1p distance.
        prior_od = G - Alpha * x_dist + pop_combine
        return prior_od

class SpatialODMAE(nn.Module):
    def __init__(self, num_static_cont, num_static_prop_multi, num_static_prop_single, num_static_zero, 
                 cont_mask_indices=None,
                 d_model=128, nhead=8, num_layers=4,use_self_loop_predictor=True):
        super().__init__()
        self.use_self_loop_predictor = use_self_loop_predictor
        self.cont_mask_indices = cont_mask_indices if cont_mask_indices is not None else []

        # X_static embeding: (B, N, F) -> (B, N, D)
        # indicator feature + 1
        total_static_features = (num_static_cont + num_static_prop_multi + num_static_prop_single + num_static_zero) + 1 
        self.static_feature_embed = nn.Sequential(
            nn.Linear(total_static_features, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        self.meta_gravity = MetaGravityPrior(d_model)
        
        # Structural OD feature embedding (GCN)
        self.od_gcn = ODGCNLayer(d_model, d_model)
        
        # Spatial Positional Encoding 
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
        
        # OOM 방지를 위해 (B, N, N, 2D)를 생성하는 MLP 대신 Bilinear 내적(Q-K) 방식 사용
        self.decoder_q = nn.Linear(d_model, d_model)
        self.decoder_k = nn.Linear(d_model, d_model)

    def forward(self, x_cont, x_prop_multi, x_prop_single, x_zero, x_od_masked, x_dist, mask, pop_raw):
        """
        x_cont, x_prop_multi, x_prop_single, x_zero: (B, N, F_x)
        x_od_masked: (B, N, N)
        x_dist: (B, N, N) distance matrix (log-scaled)
        mask: (B, N) boolean mask where True means masked (predict this)
        pop_raw: (B, N) log1p scaled raw population
        """
        B, N, _ = x_cont.shape
        
        # masking 되어야하는 feature 마스킹(종사자, 사업체 관련)
        x_cont_masked = x_cont.clone()
        
        if mask.any() and len(self.cont_mask_indices) > 0:
            # mask: (B, N) -> (B, N)
            mask_expand = mask # We can just index directly
            for idx in self.cont_mask_indices:
                x_cont_masked[..., idx] = x_cont_masked[..., idx] * (~mask_expand)
                
        # boolean mask를 float로 변환하여 indicator feature로 사용 (글로벌 OD 마스크)
        indicator = mask.float().unsqueeze(-1) # (B, N, 1)
        
        # Concat all features + global OD indicator
        x_static = torch.cat([
            x_cont_masked, x_prop_multi, x_prop_single, x_zero,
            indicator
        ], dim=-1)
        feat_emb = self.static_feature_embed(x_static) # (B, N, D)
        
        # Physics Prior Calculation (Meta-Gravity)
        prior_od = self.meta_gravity(feat_emb, x_dist, pop_raw)
        
        # Structural Features 추출
        # mask에서 True가 마스킹이므로 관측된것은 False, observed_1d는 반대
        observed_1d = ~mask
        observed_mask_2d = observed_1d.unsqueeze(1) & observed_1d.unsqueeze(2) # (B, N, N)
        
        # 마스킹된 노드는 x_od_masked에 연결이 끊겨 있으므로 메시지를 받지 못함
        od_emb = self.od_gcn(x_od_masked, feat_emb, observed_mask_2d) 
        
        # Spatial Positional Encoding 
        x_dist_eff = x_dist.clone()
        idx = torch.arange(N, device=x_dist.device)
        x_dist_eff[:, idx, idx] = 0.5
        
        A_dist = torch.exp(- x_dist_eff / (self.distance_scale ** 2 + 1e-5))
        deg_dist = A_dist.sum(dim=-1, keepdim=True) + 1e-5
        A_dist_norm = A_dist / deg_dist
        spe = self.spe_proj(torch.bmm(A_dist_norm, feat_emb))
        
        # 결합
        x = feat_emb + od_emb + spe
        
        x = self.transformer(x) # (B, N, D)
        
        # --- Pairwise OD Decoding (Memory Efficient) ---
        q = self.decoder_q(x) # (B, N, D)
        k = self.decoder_k(x) # (B, N, D)
        
        # (B, N, D) @ (B, D, N) -> (B, N, N)
        pred_od_res = torch.bmm(q, k.transpose(1, 2)) / (q.size(-1) ** 0.5)
                
        if self.use_self_loop_predictor:
            self_loop_pred = self.self_loop_predictor(x).squeeze(-1)
        else:
            self_loop_pred = 0

        b_idx = torch.arange(B).unsqueeze(-1)
        n_idx = torch.arange(N).unsqueeze(0)
        pred_od_res[b_idx, n_idx, n_idx] += self_loop_pred
        
        # Final OD is physics prior + transformer residual
        final_od = torch.nn.functional.softplus(prior_od + pred_od_res)
        prior_od_softplus = torch.nn.functional.softplus(prior_od)
        
        return final_od, prior_od_softplus

