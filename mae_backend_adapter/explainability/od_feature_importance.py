from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch


def _clone_inputs(inputs, **changes):
    """
    ModelInputs가 frozen dataclass이므로
    setattr 대신 dataclasses.replace()를 사용해
    새로운 객체를 만든다.
    """
    return replace(inputs, **changes)


def _od_prediction_change_pct(
    baseline_matrix: torch.Tensor,
    perturbed_matrix: torch.Tensor,
    o_idx: int,
    d_idx: int,
) -> float:
    """
    특정 OD Pair(출발지->도착지)의 상대 L1 변화율(%).
    """
    base_val = float(baseline_matrix[o_idx, d_idx].item())
    pert_val = float(perturbed_matrix[o_idx, d_idx].item())
    
    diff = abs(base_val - pert_val)
    denom = abs(base_val)
    if denom <= 0:
        return 0.0
    return (diff / denom) * 100.0


def get_od_feature_importance(
    *,
    predictor,
    inputs,
    baseline_matrix: torch.Tensor,
    city_codes: list[str],
    od_pairs: list[tuple[str, str]],
    feature_names: list[str] | tuple[str, ...],
    top_k: int = 5,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """
    D. 특정 통행량(OD Pair)에 대한 출발지/도착지 피처 중요도 분석.
    
    방법
    ----
    od_pairs에 등장하는 모든 고유 행정동을 한 번씩 순회하며
    해당 동의 피처를 0으로 껐을 때(Perturbation)
    관여된 통행량이 어떻게 변하는지 양방향(Origin/Destination)으로 기록한다.
    """
    
    city_codes_str = [str(code) for code in city_codes]
    feature_names = list(feature_names)
    code_to_idx = {code: idx for idx, code in enumerate(city_codes_str)}
    
    unique_nodes = set()
    for o, d in od_pairs:
        if o in code_to_idx and d in code_to_idx:
            unique_nodes.add(o)
            unique_nodes.add(d)
            
    # Pair별 저장소
    pair_results: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for o, d in od_pairs:
        if o not in code_to_idx or d not in code_to_idx:
            continue
        pair_key = f"{o}->{d}"
        pair_results[pair_key] = {
            "origin_features": [],
            "destination_features": [],
        }
        
    print()
    print(f"[D] OD Feature Importance 계산: 관련된 고유 행정동 {len(unique_nodes)}개 섭동 시작")
    
    for node_code in unique_nodes:
        node_idx = code_to_idx[node_code]
        print(f"   [{node_code}] 피처 분석 중...")
        
        for feature_idx, feature_name in enumerate(feature_names):
            perturbed_static = inputs.x_static.clone()
            original_scaled_value = float(perturbed_static[node_idx, feature_idx].item())
            
            # 특정 피처를 0으로 변경
            perturbed_static[node_idx, feature_idx] = 0.0
            perturbed_inputs = _clone_inputs(inputs, x_static=perturbed_static)
            
            (perturbed_matrix, _, _, _, _) = predictor._run_full(
                perturbed_inputs, request_metadata=None
            )
            perturbed_matrix = perturbed_matrix.float()
            
            # 이 노드가 출발지이거나 도착지인 모든 OD Pair에 대해 변화량 기록
            for o, d in od_pairs:
                if o not in code_to_idx or d not in code_to_idx:
                    continue
                    
                if node_code == o or node_code == d:
                    o_idx = code_to_idx[o]
                    d_idx = code_to_idx[d]
                    change_pct = _od_prediction_change_pct(
                        baseline_matrix, perturbed_matrix, o_idx, d_idx
                    )
                    
                    item = {
                        "feature": feature_name,
                        "feature_index": feature_idx,
                        "original_scaled_value": original_scaled_value,
                        "perturbation_scaled_value": 0.0,
                        "prediction_change_pct": change_pct,
                    }
                    
                    pair_key = f"{o}->{d}"
                    if node_code == o: # 섭동된 노드가 해당 통행량의 출발지일 경우
                        # dict의 리스트는 참조로 추가되므로 복사할 필요 없음 (다만 같은 아이템이 양쪽에 추가되면 참조가 같아지니 얕은 복사)
                        pair_results[pair_key]["origin_features"].append(dict(item))
                    if node_code == d: # 섭동된 노드가 해당 통행량의 도착지일 경우
                        pair_results[pair_key]["destination_features"].append(dict(item))
                        
    # 정렬 및 Top-K 추출
    final_results = {}
    for pair_key, data in pair_results.items():
        o_feats = data["origin_features"]
        d_feats = data["destination_features"]
        
        o_feats.sort(key=lambda x: x["prediction_change_pct"], reverse=True)
        d_feats.sort(key=lambda x: x["prediction_change_pct"], reverse=True)
        
        o_total = sum(x["prediction_change_pct"] for x in o_feats)
        d_total = sum(x["prediction_change_pct"] for x in d_feats)
        
        for rank, x in enumerate(o_feats, start=1):
            x["rank"] = rank
            x["importance_share_pct"] = (x["prediction_change_pct"] / o_total * 100.0) if o_total > 0 else 0.0
            
        for rank, x in enumerate(d_feats, start=1):
            x["rank"] = rank
            x["importance_share_pct"] = (x["prediction_change_pct"] / d_total * 100.0) if d_total > 0 else 0.0
            
        final_results[pair_key] = {
            "origin_features": o_feats[:top_k],
            "destination_features": d_feats[:top_k],
        }
        
    print(f"[D] 모든 고유 행정동 피처 섭동 및 개별 통행량 중요도 매핑 완료")
    return final_results
