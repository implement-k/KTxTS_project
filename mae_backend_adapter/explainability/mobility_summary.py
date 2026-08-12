from __future__ import annotations

from typing import Any

import torch


def get_mobility_summary(
    predicted_od: torch.Tensor,
    city_codes: list[str],
    target_codes: list[str],
    *,
    top_k: int = 20,
) -> dict[str, dict[str, Any]]:
    """
    신도시 각 가상행정동의 이동 요약을 계산한다.

    구분
    ----
    1. 같은 행정동 내부 이동
       예: 교산1동 -> 교산1동

    2. 신도시 내 다른 동과의 이동
       예:
       교산1동 -> 교산2동
       교산1동 -> 교산3동
       교산1동 -> 교산4동

    3. 신도시 외부 이동
       신도시 가상행정동을 제외한 기존 행정동과의 이동

    유출 구성:
        self + 신도시 내 다른 동 + 외부 = 전체 유출

    유입 구성:
        self + 신도시 내 다른 동에서 유입 + 외부 유입 = 전체 유입
    """

    city_codes = [
        str(code)
        for code in city_codes
    ]

    target_codes = [
        str(code)
        for code in target_codes
    ]

    n = len(city_codes)

    if tuple(predicted_od.shape) != (n, n):
        raise ValueError(
            f"predicted_od shape 오류: "
            f"{tuple(predicted_od.shape)}"
        )

    code_to_idx = {
        code: idx
        for idx, code
        in enumerate(city_codes)
    }

    target_set = set(target_codes)

    # 신도시 가상행정동이 아닌 기존 행정동
    external_indices = [
        idx
        for idx, code
        in enumerate(city_codes)
        if code not in target_set
    ]

    result: dict[str, dict[str, Any]] = {}

    for target_code in target_codes:

        if target_code not in code_to_idx:
            result[target_code] = {}
            continue

        idx = code_to_idx[target_code]

        # ====================================================
        # 전체 유출 / 전체 유입
        # ====================================================

        total_outflow = float(
            predicted_od[
                idx,
                :
            ].sum().item()
        )

        total_inflow = float(
            predicted_od[
                :,
                idx
            ].sum().item()
        )

        # ====================================================
        # 같은 행정동 내부 이동
        # ====================================================

        self_flow = float(
            predicted_od[
                idx,
                idx
            ].item()
        )

        self_outflow_ratio = (
            self_flow / total_outflow
            if total_outflow > 0
            else 0.0
        )

        self_inflow_ratio = (
            self_flow / total_inflow
            if total_inflow > 0
            else 0.0
        )

        # ====================================================
        # 신도시 내 다른 동으로 이동
        # ====================================================

        internal_outgoing = []

        for other_code in target_codes:

            if other_code == target_code:
                continue

            if other_code not in code_to_idx:
                continue

            other_idx = (
                code_to_idx[
                    other_code
                ]
            )

            trips = float(
                predicted_od[
                    idx,
                    other_idx
                ].item()
            )

            share = (
                trips / total_outflow
                if total_outflow > 0
                else 0.0
            )

            internal_outgoing.append(
                {
                    "origin_code":
                        target_code,

                    "destination_code":
                        other_code,

                    "trips":
                        trips,

                    "share":
                        share,

                    "share_pct":
                        share * 100,
                }
            )

        internal_outgoing.sort(
            key=lambda x: x["trips"],
            reverse=True,
        )

        # ====================================================
        # 신도시 내 다른 동에서 유입
        # ====================================================

        internal_incoming = []

        for other_code in target_codes:

            if other_code == target_code:
                continue

            if other_code not in code_to_idx:
                continue

            other_idx = (
                code_to_idx[
                    other_code
                ]
            )

            trips = float(
                predicted_od[
                    other_idx,
                    idx
                ].item()
            )

            share = (
                trips / total_inflow
                if total_inflow > 0
                else 0.0
            )

            internal_incoming.append(
                {
                    "origin_code":
                        other_code,

                    "destination_code":
                        target_code,

                    "trips":
                        trips,

                    "share":
                        share,

                    "share_pct":
                        share * 100,
                }
            )

        internal_incoming.sort(
            key=lambda x: x["trips"],
            reverse=True,
        )

        # ====================================================
        # 신도시 내 다른 동 이동 합계
        # ====================================================

        internal_outflow = sum(
            item["trips"]
            for item in internal_outgoing
        )

        internal_inflow = sum(
            item["trips"]
            for item in internal_incoming
        )

        internal_outflow_ratio = (
            internal_outflow
            / total_outflow
            if total_outflow > 0
            else 0.0
        )

        internal_inflow_ratio = (
            internal_inflow
            / total_inflow
            if total_inflow > 0
            else 0.0
        )

        # ====================================================
        # 신도시 외부 이동
        # ====================================================

        external_outflow = float(
            predicted_od[
                idx,
                external_indices
            ].sum().item()
        )

        external_inflow = float(
            predicted_od[
                external_indices,
                idx
            ].sum().item()
        )

        external_outflow_ratio = (
            external_outflow
            / total_outflow
            if total_outflow > 0
            else 0.0
        )

        external_inflow_ratio = (
            external_inflow
            / total_inflow
            if total_inflow > 0
            else 0.0
        )

        # ====================================================
        # 검증용 합계
        # ====================================================

        outflow_check = (
            self_flow
            + internal_outflow
            + external_outflow
        )

        inflow_check = (
            self_flow
            + internal_inflow
            + external_inflow
        )

        outflow_ratio_check = (
            self_outflow_ratio
            + internal_outflow_ratio
            + external_outflow_ratio
        )

        inflow_ratio_check = (
            self_inflow_ratio
            + internal_inflow_ratio
            + external_inflow_ratio
        )

        # ====================================================
        # 외부로 많이 가는 행정동 TOP-K
        # ====================================================

        outgoing_values = (
            predicted_od[
                idx,
                external_indices
            ]
            .float()
        )

        outgoing_k = min(
            top_k,
            len(external_indices),
        )

        top_out_values, top_out_positions = (
            torch.topk(
                outgoing_values,
                k=outgoing_k,
            )
        )

        top_outgoing = []

        for value, relative_pos in zip(
            top_out_values.tolist(),
            top_out_positions.tolist(),
        ):

            real_idx = (
                external_indices[
                    relative_pos
                ]
            )

            code = city_codes[
                real_idx
            ]

            trips = float(value)

            share = (
                trips / total_outflow
                if total_outflow > 0
                else 0.0
            )

            top_outgoing.append(
                {
                    "dong_code":
                        code,

                    "trips":
                        trips,

                    "share":
                        share,

                    "share_pct":
                        share * 100,
                }
            )

        # ====================================================
        # 외부에서 많이 오는 행정동 TOP-K
        # ====================================================

        incoming_values = (
            predicted_od[
                external_indices,
                idx
            ]
            .float()
        )

        incoming_k = min(
            top_k,
            len(external_indices),
        )

        top_in_values, top_in_positions = (
            torch.topk(
                incoming_values,
                k=incoming_k,
            )
        )

        top_incoming = []

        for value, relative_pos in zip(
            top_in_values.tolist(),
            top_in_positions.tolist(),
        ):

            real_idx = (
                external_indices[
                    relative_pos
                ]
            )

            code = city_codes[
                real_idx
            ]

            trips = float(value)

            share = (
                trips / total_inflow
                if total_inflow > 0
                else 0.0
            )

            top_incoming.append(
                {
                    "dong_code":
                        code,

                    "trips":
                        trips,

                    "share":
                        share,

                    "share_pct":
                        share * 100,
                }
            )

        # ====================================================
        # 최종 반환
        # ====================================================

        result[target_code] = {

            "dong_code":
                target_code,

            # 전체
            "total_outflow":
                total_outflow,

            "total_inflow":
                total_inflow,

            # 같은 동 내부 이동
            "self_flow":
                self_flow,

            "self_outflow_ratio":
                self_outflow_ratio,

            "self_outflow_ratio_pct":
                self_outflow_ratio * 100,

            "self_inflow_ratio":
                self_inflow_ratio,

            "self_inflow_ratio_pct":
                self_inflow_ratio * 100,

            # 신도시 내 다른 동
            "internal_outgoing":
                internal_outgoing,

            "internal_incoming":
                internal_incoming,

            "internal_outflow":
                internal_outflow,

            "internal_outflow_ratio":
                internal_outflow_ratio,

            "internal_outflow_ratio_pct":
                internal_outflow_ratio * 100,

            "internal_inflow":
                internal_inflow,

            "internal_inflow_ratio":
                internal_inflow_ratio,

            "internal_inflow_ratio_pct":
                internal_inflow_ratio * 100,

            # 외부
            "external_outflow":
                external_outflow,

            "external_outflow_ratio":
                external_outflow_ratio,

            "external_outflow_ratio_pct":
                external_outflow_ratio * 100,

            "external_inflow":
                external_inflow,

            "external_inflow_ratio":
                external_inflow_ratio,

            "external_inflow_ratio_pct":
                external_inflow_ratio * 100,

            # 합계 검증
            "outflow_check":
                outflow_check,

            "inflow_check":
                inflow_check,

            "outflow_ratio_check_pct":
                outflow_ratio_check * 100,

            "inflow_ratio_check_pct":
                inflow_ratio_check * 100,

            # 외부 TOP-K
            "top_outgoing":
                top_outgoing,

            "top_incoming":
                top_incoming,
        }

    return result