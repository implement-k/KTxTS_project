import torch
import torch.nn as nn

class SpatialODMAE(nn.Module):
    def __init__(self, num_nodes, num_features=13, d_model=128, nhead=8, num_layers=4):
        super().__init__()
        # X_static embeding: (B, N, F) -> (B, N, D) - leanable
        # OD feature embedding: (B, N, 2N) -> (B, N, D) - leanable
        self.feature_embed = nn.Sequential(
            nn.Linear(num_features, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.od_embed = nn.Linear(num_nodes * 2, d_model)
        
        # OD 정보의 반영 비율을 조절하는 Learnable Gating Network
        self.od_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
        # distance based 상대 positional bias
        self.nhead = nhead
        self.distance_bias = nn.Embedding(50, nhead)
        # 0부터 5.5 구간을 49개로 나눔
        self.register_buffer('boundaries', torch.linspace(0, 5.5, 49))
        
        # Mask Token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # decoder: (B, N, D) -> (B, N, 2N)
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
            
        # Note: Since x_od_masked is symmetric conceptually in how we gather, we just transpose
        row_feat = x_od_masked
        col_feat = x_od_masked.transpose(1, 2)
        node_od_feat = torch.cat([row_feat, col_feat], dim=-1) # (B, N, 2N)
        
        # feat_emb: (B, N, D), od_emb: (B, N, D) - 임베딩
        feat_emb = self.feature_embed(x_static) 
        od_emb = self.od_embed(node_od_feat)   
        
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
        distance_bins = torch.bucketize(x_dist, self.boundaries) # (B, N, N)
        
        # bias: (B, N, N, nhead) - 각 distance bin에 대해 nhead 차원의 bias를 가져옴
        bias = self.distance_bias(distance_bins) 
        
        # bias: (B, N, N, nhead) -> (B * nhead, N, N)
        bias = bias.permute(0, 3, 1, 2).reshape(B * self.nhead, N, N)
        
        # Transformer (bias 적용)
        x = self.transformer(x, mask=bias) # (B, N, D)
        
        # out: (B, N, 2N) - decode
        out = self.decoder(x) 
        
        # Average
        pred_od = (out[..., :N] + out[..., N:].transpose(1, 2)) / 2.0
        
        return pred_od
