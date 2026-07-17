import torch
import torch.nn as nn

class SpatialODMAE(nn.Module):
    def __init__(self, num_nodes, num_features=13, d_model=128, nhead=8, num_layers=4):
        super().__init__()
        # X_static embeding: (B, N, F) -> (B, N, D) - leanable
        # OD feature embedding: (B, N, 2N) -> (B, N, D) - leanable
        self.feature_embed = nn.Linear(num_features, d_model)
        self.od_embed = nn.Linear(num_nodes * 2, d_model)
        
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
