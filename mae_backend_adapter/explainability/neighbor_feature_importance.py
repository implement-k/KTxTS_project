from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch


def _clone_inputs(
    inputs,
    **changes,
):
    """
    ModelInputs가 frozen dataclass이므로
    setattr로 수정하지 않고 dataclasses.replace()로
    새로운 객체를 생성한다.
    """
    return replace(
        inputs,
        **changes,
    )


def _target_prediction_vector(
    matrix: torch.Tensor,
    target_idx: int,
) -> torch.Tensor:
    """
    해당 행정동과 관련된 전체 OD를 하나의 벡터로 만든다.
    """
    outgoing = matrix[target_idx, :].float()
    incoming_before = matrix[:target_idx, target_idx].float()
    incoming_after = matrix[target_idx + 1:, target_idx].float()
    incoming = torch.cat([incoming_before, incoming_after])
    
    return torch.cat([outgoing, incoming])


def _prediction_change_pct(
    baseline_matrix: torch.Tensor,
    perturbed_matrix: torch.Tensor,
    target_idx: int,
) -> float:
    """
    Perturbation 전/후 해당 동 OD의 상대 L1 변화율(%).
    """
    baseline = _target_prediction_vector(baseline_matrix, target_idx)
    perturbed = _target_prediction_vector(perturbed_matrix, target_idx)

    denominator = float(torch.abs(baseline).sum().item())
    if denominator <= 0:
        return 0.0

    difference = float(torch.abs(baseline - perturbed).sum().item())
    return (difference / denominator) * 100.0


def get_neighbor_feature_importance(
    *,
    predictor,
    inputs,
    baseline_matrix: torch.Tensor,
    city_codes: list[str],
    target_codes: list[str],
    neighbors_dict: dict[str, list[str]],
    feature_names: list[str] | tuple[str, ...],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """
    이웃 행정동의 특정 피처가 Target 동의 예측치에 미치는 영향을 분석한다.
    
    방법
    ----
    Target 동의 각 이웃에 대해, 이웃의 static feature를 하나씩 0으로 바꾼 뒤
    MAE를 재실행하여 Target 동의 OD(유출입 벡터)가 얼마나 변했는지 계산한다.
    """

    city_codes = [str(code) for code in city_codes]
    target_codes = [str(code) for code in target_codes]
    feature_names = list(feature_names)
    
    code_to_idx = {code: idx for idx, code in enumerate(city_codes)}
    feature_count = len(feature_names)

    if inputs.x_static.shape[1] < feature_count:
        raise ValueError(
            "x_static feature 차원이 feature_names보다 작습니다.\n"
            f"x_static shape = {tuple(inputs.x_static.shape)}\n"
            f"feature_names = {feature_count}"
        )

    result: dict[str, dict[str, list[dict[str, Any]]]] = {}

    for target_code in target_codes:
        if target_code not in code_to_idx:
            result[target_code] = {}
            continue

        target_idx = code_to_idx[target_code]
        result[target_code] = {}
        
        target_neighbors = [str(c) for c in neighbors_dict.get(target_code, [])]

        print()
        print(f"[C] Neighbor Feature Importance 계산: {target_code}")
        
        for neighbor_code in target_neighbors:
            if neighbor_code not in code_to_idx:
                continue
                
            neighbor_idx = code_to_idx[neighbor_code]
            neighbor_items = []
            
            print(f"   [{target_code}] 이웃 {neighbor_code}의 피처 분석 중...")

            for feature_idx, feature_name in enumerate(feature_names):
                perturbed_static = inputs.x_static.clone()
                original_scaled_value = float(perturbed_static[neighbor_idx, feature_idx].item())
                
                # 이웃 동네의 특정 피처를 scaled 0으로 변경
                perturbed_static[neighbor_idx, feature_idx] = 0.0
                
                perturbed_inputs = _clone_inputs(inputs, x_static=perturbed_static)
                
                (perturbed_matrix, _metadata, _origins, _destinations, _zones) = predictor._run_full(
                    perturbed_inputs, request_metadata=None
                )
                
                perturbed_matrix = perturbed_matrix.float()
                
                # Baseline(원본) 대비 Target 동네의 변화율 계산
                change_pct = _prediction_change_pct(
                    baseline_matrix=baseline_matrix,
                    perturbed_matrix=perturbed_matrix,
                    target_idx=target_idx,
                )
                
                neighbor_items.append({
                    "feature": feature_name,
                    "feature_index": feature_idx,
                    "original_scaled_value": original_scaled_value,
                    "perturbation_scaled_value": 0.0,
                    "prediction_change_pct": change_pct,
                })
                
            # 정렬 및 상대 중요도 계산
            neighbor_items.sort(key=lambda x: x["prediction_change_pct"], reverse=True)
            total_impact = sum(item["prediction_change_pct"] for item in neighbor_items)
            
            for rank, item in enumerate(neighbor_items, start=1):
                item["rank"] = rank
                item["importance_share_pct"] = (
                    (item["prediction_change_pct"] / total_impact * 100.0) 
                    if total_impact > 0 else 0.0
                )
                
            result[target_code][neighbor_code] = neighbor_items

        print(f"[C] {target_code} 완료")

    return result
