import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import ODMAE

def test_counterfactual():
    batch_size = 2
    num_nodes = 50
    static_dim = 15
    d_model = 128
    
    # 1. 모델 초기화
    print("Initializing Hybrid ODMAE model...")
    model = ODMAE(num_features=static_dim, d_model=d_model, nhead=4, num_layers=2)
    model.eval()
    
    # 2. 더미 데이터 생성
    x_static = torch.randn(batch_size, num_nodes, static_dim)
    x_od_masked = torch.randn(batch_size, num_nodes, num_nodes)
    x_dist = torch.randn(batch_size, num_nodes, num_nodes)
    A_spatial = torch.randn(batch_size, num_nodes, num_nodes)
    mask = torch.zeros(batch_size, num_nodes, dtype=torch.bool)
    # 임의로 몇 개의 노드를 마스킹 (예: 신도시)
    mask[:, 5:10] = True 
    active_node_mask = torch.ones(batch_size, num_nodes, dtype=torch.bool)
    
    # 3. Base 시나리오 (가중치 조절 없음)
    print("\n--- Running Base Scenario ---")
    with torch.no_grad():
        base_logit = model(x_static, x_od_masked, x_dist, A_spatial, mask, active_node_mask)
        
    print(f"Base output shape: {base_logit.shape}")
    
    # 4. Counterfactual 시나리오 (예: 특정 Feature의 중요도 2배 증가)
    # static feature 0번이 '인구(Population)'라고 가정해봅시다.
    print("\n--- Running Counterfactual Scenario (Population Importance x 2) ---")
    
    O_adj = torch.ones(static_dim)
    D_adj = torch.ones(static_dim)
    
    # 0번 피처(인구)의 출발/도착 잠재력 가중치를 2.0배로 뻥튀기!
    O_adj[0] = 2.0
    D_adj[0] = 2.0
    
    with torch.no_grad():
        cf_logit = model(
            x_static, x_od_masked, x_dist, A_spatial, mask, active_node_mask,
            O_importance_adj=O_adj, D_importance_adj=D_adj
        )
        
    # 비교: 특정 노드(마스킹된 5번 노드)의 전체 통행량 변화 확인
    base_flow = torch.expm1(base_logit[0, 5, :]).sum().item()
    cf_flow = torch.expm1(cf_logit[0, 5, :]).sum().item()
    
    print(f"Node 5 Total Outgoing Flow (Base) : {base_flow:.4f}")
    print(f"Node 5 Total Outgoing Flow (CF)   : {cf_flow:.4f}")
    
    diff = cf_flow - base_flow
    print(f"\nCounterfactual Difference: {diff:+.4f} (인구 중요도 조절이 물리적 거시 모델(Gravity)을 통해 전체 예측값에 직접 반영됨!)")

if __name__ == "__main__":
    test_counterfactual()
