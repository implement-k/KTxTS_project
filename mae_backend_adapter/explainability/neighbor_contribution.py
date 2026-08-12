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
    setattr 대신 dataclasses.replace()를 사용해
    새로운 객체를 만든다.
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
    해당 target 동과 관련된 전체 OD 벡터.

    포함:
    - target -> 모든 동
    - 모든 동 -> target

    self OD는 중복을 막기 위해
    incoming 벡터에서는 제외한다.
    """

    outgoing = matrix[
        target_idx,
        :
    ].float()

    incoming_before = matrix[
        :target_idx,
        target_idx,
    ].float()

    incoming_after = matrix[
        target_idx + 1:,
        target_idx,
    ].float()

    incoming = torch.cat(
        [
            incoming_before,
            incoming_after,
        ]
    )

    return torch.cat(
        [
            outgoing,
            incoming,
        ]
    )


def _prediction_change_pct(
    baseline_matrix: torch.Tensor,
    perturbed_matrix: torch.Tensor,
    target_idx: int,
) -> float:
    """
    adjacency 연결 제거 전후
    target 동 OD의 상대 L1 변화율(%).
    """

    baseline = _target_prediction_vector(
        baseline_matrix,
        target_idx,
    )

    perturbed = _target_prediction_vector(
        perturbed_matrix,
        target_idx,
    )

    denominator = float(
        torch.abs(
            baseline
        ).sum().item()
    )

    if denominator <= 0:
        return 0.0

    difference = float(
        torch.abs(
            baseline
            - perturbed
        ).sum().item()
    )

    return (
        difference
        / denominator
        * 100.0
    )


def get_neighbor_contributions(
    *,
    predictor,
    inputs,
    baseline_matrix: torch.Tensor,
    city_codes: list[str],
    target_codes: list[str],
    neighbors: dict[str, list[str]],
) -> dict[str, list[dict[str, Any]]]:
    """
    B. 이웃 행정동이 해당 동 예측에 얼마나 영향을 줬는지 계산.

    방법
    ----
    target 동과 특정 neighbor 사이의 adjacency 연결을
    양방향 모두 제거한다.

        target -> neighbor = 0
        neighbor -> target = 0

    이후 MAE를 다시 실행하여
    baseline OD 대비 target 동의 OD가 얼마나 변했는지 계산한다.

    prediction_change_pct:
        해당 이웃 연결 제거로 예측이 변한 정도.

    contribution_share_pct:
        모든 이웃 perturbation 영향 중
        해당 이웃이 차지하는 상대적 비중.
    """

    city_codes = [
        str(code)
        for code in city_codes
    ]

    target_codes = [
        str(code)
        for code in target_codes
    ]

    code_to_idx = {
        code: idx
        for idx, code in enumerate(
            city_codes
        )
    }

    result: dict[
        str,
        list[dict[str, Any]]
    ] = {}

    # ========================================================
    # 신도시 각 동
    # ========================================================

    for target_code in target_codes:

        if target_code not in code_to_idx:

            result[
                target_code
            ] = []

            continue

        target_idx = code_to_idx[
            target_code
        ]

        target_neighbors = [
            str(code)
            for code in neighbors.get(
                target_code,
                [],
            )
        ]

        items = []

        print()
        print(
            f"[B] Neighbor contribution 계산: "
            f"{target_code}"
        )

        # ====================================================
        # 이웃 하나씩 제거
        # ====================================================

        for neighbor_order, neighbor_code in enumerate(
            target_neighbors,
            start=1,
        ):

            if neighbor_code not in code_to_idx:
                continue

            neighbor_idx = (
                code_to_idx[
                    neighbor_code
                ]
            )

            print(
                f"   [{target_code}] "
                f"{neighbor_order}/"
                f"{len(target_neighbors)} "
                f"이웃 {neighbor_code} 계산 중..."
            )

            # ------------------------------------------------
            # adjacency 복사
            # ------------------------------------------------

            perturbed_adj = (
                inputs.a_spatial
                .clone()
            )

            original_forward = float(
                perturbed_adj[
                    target_idx,
                    neighbor_idx,
                ].item()
            )

            original_backward = float(
                perturbed_adj[
                    neighbor_idx,
                    target_idx,
                ].item()
            )

            # ------------------------------------------------
            # 양방향 adjacency 연결 제거
            # ------------------------------------------------

            perturbed_adj[
                target_idx,
                neighbor_idx,
            ] = 0.0

            perturbed_adj[
                neighbor_idx,
                target_idx,
            ] = 0.0

            # ------------------------------------------------
            # frozen dataclass -> replace
            # ------------------------------------------------

            perturbed_inputs = _clone_inputs(
                inputs,
                a_spatial=
                    perturbed_adj,
            )

            # ------------------------------------------------
            # MAE 재예측
            # ------------------------------------------------

            (
                perturbed_matrix,
                _metadata,
                _origins,
                _destinations,
                _zones,
            ) = predictor._run_full(
                perturbed_inputs,
                request_metadata=None,
            )

            perturbed_matrix = (
                perturbed_matrix.float()
            )

            # ------------------------------------------------
            # Baseline 대비 변화
            # ------------------------------------------------

            change_pct = (
                _prediction_change_pct(
                    baseline_matrix=
                        baseline_matrix,

                    perturbed_matrix=
                        perturbed_matrix,

                    target_idx=
                        target_idx,
                )
            )

            print(
                f"      완료 "
                f"| 예측 변화율 "
                f"{change_pct:.6f}%"
            )

            items.append(
                {
                    "dong_code":
                        neighbor_code,

                    "prediction_change_pct":
                        change_pct,

                    "original_edge_target_to_neighbor":
                        original_forward,

                    "original_edge_neighbor_to_target":
                        original_backward,
                }
            )

        # ====================================================
        # 영향도 높은 이웃 순 정렬
        # ====================================================

        items.sort(
            key=lambda x:
                x[
                    "prediction_change_pct"
                ],
            reverse=True,
        )

        # ====================================================
        # 상대 기여도
        # ====================================================

        total_impact = sum(
            item[
                "prediction_change_pct"
            ]
            for item in items
        )

        for rank, item in enumerate(
            items,
            start=1,
        ):

            item[
                "rank"
            ] = rank

            if total_impact > 0:

                contribution_share = (
                    item[
                        "prediction_change_pct"
                    ]
                    / total_impact
                    * 100.0
                )

            else:

                contribution_share = 0.0

            item[
                "contribution_share_pct"
            ] = contribution_share

        result[
            target_code
        ] = items

        print(
            f"[B] {target_code} 완료"
        )

    return result