from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


# ============================================================
# 웹 표시용 이름
# ============================================================

FEATURE_LABELS = {
    "행정동전체면적_m2":
        "행정동 면적",

    "주거지역비율_pct":
        "주거지역 비율",

    "상업업무지역비율_pct":
        "상업·업무지역 비율",

    "공공시설지역비율_pct":
        "공공시설지역 비율",

    "기타지역비율_pct":
        "기타지역 비율",

    "아파트비율_퍼센트":
        "아파트 비율",

    "pop_0_19":
        "0~19세 인구",

    "pop_20_59":
        "20~59세 인구",

    "pop_60_plus":
        "60세 이상 인구",

    "worker_count":
        "종사자 수",

    "business_count":
        "사업체 수",

    "worker_density":
        "종사자 밀도",

    "business_density":
        "사업체 밀도",

    "station_count_지하철":
        "지하철역 수",

    "station_density_지하철":
        "지하철역 밀도",

    "station_count_일반철도":
        "일반철도역 수",

    "station_count_준고속철도":
        "준고속철도역 수",

    "station_count_고속철도":
        "고속철도역 수",
}


# 신도시에서 현재 시나리오 생성 시
# 의도적으로 0 초기화되는 변수
ZERO_INITIALIZED_FEATURES = {
    "worker_count",
    "business_count",
    "worker_density",
    "business_density",
}


def _format_value(
    feature_name: str,
    value: float,
) -> str:
    """
    웹/터미널 표시용 원본 값.
    """

    if (
        feature_name.endswith("_pct")
        or
        feature_name
        == "아파트비율_퍼센트"
    ):
        return f"{value:.1f}%"

    if (
        feature_name
        == "행정동전체면적_m2"
    ):
        return (
            f"{value / 1_000_000:.2f} km²"
        )

    if feature_name in {
        "pop_0_19",
        "pop_20_59",
        "pop_60_plus",
        "worker_count",
        "business_count",
    }:
        return f"{value:,.0f}"

    if feature_name.startswith(
        "station_count_"
    ):
        return f"{value:,.0f}개"

    if (
        "density"
        in feature_name
    ):
        # 기존 0.0000 문제 방지
        return f"{value:.6f}"

    return f"{value:.4f}"


def get_similar_dongs_by_static(
    x_static: torch.Tensor,
    x_static_raw: torch.Tensor,
    feature_names: list[str] | tuple[str, ...],
    city_codes: list[str],
    target_codes: list[str],
    *,
    top_k: int = 5,
    reason_top_k: int = 5,
    exclude_target_codes: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    """
    전체 도시특성 유사동:
        scaled 18개 static feature cosine similarity.

    '왜 비슷한가':
        feature별 standardized gap이 작은 특성을 반환.

    상세 설명에서는
    - 0 == 0
    - 신도시에서 의도적으로 0 초기화된 worker/business 관련 변수
    를 제외한다.
    """

    if x_static.ndim != 2:
        raise ValueError(
            "x_static shape 오류"
        )

    if (
        x_static_raw.shape
        != x_static.shape
    ):
        raise ValueError(
            "x_static_raw shape 오류"
        )

    if (
        len(feature_names)
        != x_static.shape[1]
    ):
        raise ValueError(
            "feature_names 수 오류"
        )

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
        for idx, code
        in enumerate(city_codes)
    }

    # ========================================================
    # 전체 도시특성 유사도
    # ========================================================

    normalized = F.normalize(
        x_static.float(),
        p=2,
        dim=1,
    )

    similarity_matrix = (
        normalized
        @ normalized.T
    )

    target_set = set(
        target_codes
    )

    result = {}

    for target_code in target_codes:

        if target_code not in code_to_idx:
            result[target_code] = []
            continue

        target_idx = (
            code_to_idx[
                target_code
            ]
        )

        scores = (
            similarity_matrix[
                target_idx
            ]
            .clone()
        )

        scores[target_idx] = float(
            "-inf"
        )

        if exclude_target_codes:

            for code in target_set:

                if code in code_to_idx:

                    scores[
                        code_to_idx[
                            code
                        ]
                    ] = float(
                        "-inf"
                    )

        valid_count = int(
            torch.isfinite(
                scores
            ).sum().item()
        )

        k = min(
            top_k,
            valid_count,
        )

        if k == 0:
            result[target_code] = []
            continue

        top_scores, top_indices = (
            torch.topk(
                scores,
                k=k,
            )
        )

        items = []

        # ====================================================
        # 후보 동마다 이유 계산
        # ====================================================

        for (
            overall_similarity,
            candidate_idx,
        ) in zip(
            top_scores.tolist(),
            top_indices.tolist(),
        ):

            candidate_code = (
                city_codes[
                    candidate_idx
                ]
            )

            reasons = []

            for (
                feature_idx,
                feature_name,
            ) in enumerate(
                feature_names
            ):

                target_raw = float(
                    x_static_raw[
                        target_idx,
                        feature_idx,
                    ]
                )

                candidate_raw = float(
                    x_static_raw[
                        candidate_idx,
                        feature_idx,
                    ]
                )

                # --------------------------------------------
                # 둘 다 0이면 설명에서 제외
                # --------------------------------------------

                if (
                    abs(target_raw)
                    < 1e-12
                    and
                    abs(candidate_raw)
                    < 1e-12
                ):
                    continue

                # --------------------------------------------
                # 신도시에서 의도적으로 0 초기화되는
                # worker/business 변수는 설명에서 제외
                # --------------------------------------------

                if (
                    feature_name
                    in ZERO_INITIALIZED_FEATURES
                    and
                    abs(target_raw)
                    < 1e-12
                ):
                    continue

                standardized_gap = abs(
                    float(
                        x_static[
                            target_idx,
                            feature_idx,
                        ]
                    )
                    -
                    float(
                        x_static[
                            candidate_idx,
                            feature_idx,
                        ]
                    )
                )

                feature_similarity = (
                    1.0
                    /
                    (
                        1.0
                        + standardized_gap
                    )
                )

                reasons.append(
                    {
                        "feature":
                            feature_name,

                        "label":
                            FEATURE_LABELS.get(
                                feature_name,
                                feature_name,
                            ),

                        "target_value":
                            target_raw,

                        "candidate_value":
                            candidate_raw,

                        "target_display":
                            _format_value(
                                feature_name,
                                target_raw,
                            ),

                        "candidate_display":
                            _format_value(
                                feature_name,
                                candidate_raw,
                            ),

                        "standardized_gap":
                            standardized_gap,

                        "feature_similarity":
                            feature_similarity,
                    }
                )

            # 가장 비슷한 피처 순
            reasons.sort(
                key=lambda x:
                    x[
                        "standardized_gap"
                    ]
            )

            reasons = reasons[
                :reason_top_k
            ]

            items.append(
                {
                    "dong_code":
                        candidate_code,

                    "similarity":
                        float(
                            overall_similarity
                        ),

                    "similar_features":
                        reasons,
                }
            )

        result[
            target_code
        ] = items

    return result