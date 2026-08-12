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

    포함:
    - target -> 모든 동
    - 모든 동 -> target

    self OD는 중복을 피하기 위해
    incoming 쪽에서는 제외한다.
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
    Perturbation 전/후 해당 동 OD의 상대 L1 변화율(%).

    0%:
        예측 변화 없음

    값이 클수록:
        해당 feature가 해당 동 OD 예측에 더 큰 영향을 준다고 해석.
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


def get_feature_importance(
    *,
    predictor,
    inputs,
    baseline_matrix: torch.Tensor,
    city_codes: list[str],
    target_codes: list[str],
    feature_names: list[str] | tuple[str, ...],
    top_k: int = 5,
) -> dict[str, list[dict[str, Any]]]:
    """
    A. 각 신도시 행정동 예측의 Feature Importance 계산.

    방법
    ----
    각 target 동에 대해 static feature를 하나씩 perturbation한다.

    현재 모델 입력의 x_static은 scaling된 feature이므로,
    해당 feature를 scaled value 0으로 바꾼다.

    그 뒤 MAE를 다시 실행하여
    baseline OD 대비 해당 동의 전체 유출/유입 OD가
    얼마나 변했는지 계산한다.

    반환
    ----
    prediction_change_pct:
        해당 feature perturbation으로 예측 OD가 변한 정도.

    importance_share_pct:
        모든 feature perturbation 영향량 중
        해당 feature가 차지하는 상대 비중.
    """

    city_codes = [
        str(code)
        for code in city_codes
    ]

    target_codes = [
        str(code)
        for code in target_codes
    ]

    feature_names = list(
        feature_names
    )

    code_to_idx = {
        code: idx
        for idx, code in enumerate(
            city_codes
        )
    }

    feature_count = len(
        feature_names
    )

    if inputs.x_static.shape[1] < feature_count:
        raise ValueError(
            "x_static feature 차원이 feature_names보다 작습니다.\n"
            f"x_static shape = {tuple(inputs.x_static.shape)}\n"
            f"feature_names = {feature_count}"
        )

    result: dict[
        str,
        list[dict[str, Any]]
    ] = {}

    # ========================================================
    # 신도시 각 동
    # ========================================================

    for target_code in target_codes:

        if target_code not in code_to_idx:
            result[target_code] = []
            continue

        target_idx = code_to_idx[target_code]
        items = []

        print()
        print(
            f"[A] Feature importance 계산: "
            f"{target_code}"
        )

        # ====================================================
        # Feature 하나씩 perturbation
        # ====================================================

        for feature_idx, feature_name in enumerate(
            feature_names
        ):

            print(
                f"   [{target_code}] "
                f"{feature_idx + 1}/{feature_count} "
                f"{feature_name} 계산 중..."
            )

            # ------------------------------------------------
            # Static Feature 복사
            # ------------------------------------------------

            perturbed_static = (
                inputs.x_static
                .clone()
            )

            original_scaled_value = float(
                perturbed_static[
                    target_idx,
                    feature_idx,
                ].item()
            )

            # ------------------------------------------------
            # 해당 Feature를 scaled 0으로 변경
            # ------------------------------------------------

            perturbed_static[
                target_idx,
                feature_idx,
            ] = 0.0

            # ------------------------------------------------
            # frozen dataclass이므로 replace() 사용
            # ------------------------------------------------

            perturbed_inputs = _clone_inputs(
                inputs,
                x_static=
                    perturbed_static,
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
            # Baseline 대비 변화율
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
                    "feature": feature_name,
                    "feature_index": feature_idx,
                    "original_scaled_value": original_scaled_value,
                    "perturbation_scaled_value": 0.0,
                    "prediction_change_pct": change_pct,
                }
            )

        # ====================================================
        # 영향도 순 정렬
        # ====================================================

        items.sort(
            key=lambda x:
                x[
                    "prediction_change_pct"
                ],
            reverse=True,
        )

        # ====================================================
        # 상대 중요도
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

                importance_share = (
                    item[
                        "prediction_change_pct"
                    ]
                    / total_impact
                    * 100.0
                )

            else:

                importance_share = 0.0

            item["importance_share_pct"] = importance_share

        result[target_code] = items[:top_k]

        print(
            f"[A] {target_code} 완료"
        )

    return result