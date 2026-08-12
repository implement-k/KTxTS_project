from __future__ import annotations

from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F

from .common import BASE_DIR


def _cosine(
    a: torch.Tensor,
    b: torch.Tensor,
) -> float:

    if (
        torch.linalg.vector_norm(
            a
        ).item() == 0
        or
        torch.linalg.vector_norm(
            b
        ).item() == 0
    ):
        return 0.0

    return float(
        F.cosine_similarity(
            a.unsqueeze(0).float(),
            b.unsqueeze(0).float(),
        ).item()
    )


def _get_common_patterns(
    target_values: torch.Tensor,
    candidate_values: torch.Tensor,
    comparison_codes: list[str],
    *,
    candidate_code: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    '왜 이동패턴이 비슷한가'에 보여줄 지역을 선택한다.

    기준:
    1. 두 동 모두에서 일정 비중 이상 나타나는가
    2. 두 동의 이동 비중이 실제로 가까운가
    3. 동시에 이동 비중 자체도 충분히 큰가

    reason_score
        = min_share * closeness

    closeness
        = 1 - |a-b| / max(a,b)

    예:
        1.11% vs 1.54% -> 높은 점수
        0.96% vs 7.95% -> 큰 차이 때문에 낮은 점수
    """

    target_values = (
        target_values.float()
    )

    candidate_values = (
        candidate_values.float()
    )

    target_total = float(
        target_values.sum().item()
    )

    candidate_total = float(
        candidate_values.sum().item()
    )

    if (
        target_total <= 0
        or
        candidate_total <= 0
    ):
        return []

    target_share = (
        target_values
        / target_total
    )

    candidate_share = (
        candidate_values
        / candidate_total
    )

    # 두 동 중 작은 비중
    min_share = torch.minimum(
        target_share,
        candidate_share,
    )

    max_share = torch.maximum(
        target_share,
        candidate_share,
    )

    difference = torch.abs(
        target_share
        - candidate_share
    )

    # 비중이 얼마나 가까운지
    closeness = torch.where(
        max_share > 0,
        1.0
        - (
            difference
            / max_share
        ),
        torch.zeros_like(
            max_share
        ),
    )

    # 큰 비중 + 비슷한 비율
    reason_score = (
        min_share
        * closeness
    )

    # 비교동 자기 자신 self OD는
    # 클릭 상세 설명에서는 제거
    if candidate_code in comparison_codes:

        self_idx = (
            comparison_codes.index(
                candidate_code
            )
        )

        reason_score[
            self_idx
        ] = -1.0

    valid_indices = torch.where(
        reason_score > 0
    )[0]

    if len(valid_indices) == 0:
        return []

    valid_scores = (
        reason_score[
            valid_indices
        ]
    )

    k = min(
        top_k,
        len(valid_indices),
    )

    top_scores, order = (
        torch.topk(
            valid_scores,
            k=k,
        )
    )

    top_indices = (
        valid_indices[
            order
        ]
    )

    result = []

    for (
        score,
        idx,
    ) in zip(
        top_scores.tolist(),
        top_indices.tolist(),
    ):

        target_share_value = float(
            target_share[
                idx
            ]
        )

        candidate_share_value = float(
            candidate_share[
                idx
            ]
        )

        result.append(
            {
                "dong_code":
                    comparison_codes[
                        idx
                    ],

                "target_trips":
                    float(
                        target_values[
                            idx
                        ]
                    ),

                "candidate_trips":
                    float(
                        candidate_values[
                            idx
                        ]
                    ),

                "target_share":
                    target_share_value,

                "candidate_share":
                    candidate_share_value,

                "target_share_pct":
                    target_share_value
                    * 100,

                "candidate_share_pct":
                    candidate_share_value
                    * 100,

                "share_gap_pct":
                    abs(
                        target_share_value
                        - candidate_share_value
                    )
                    * 100,

                "reason_score":
                    float(score),
            }
        )

    return result


def get_similar_dongs_by_od(
    predicted_od: torch.Tensor,
    actual_od: torch.Tensor,
    city_codes: list[str],
    target_codes: list[str],
    *,
    top_k: int = 5,
    reason_top_k: int = 5,
) -> dict[str, list[dict[str, Any]]]:

    city_codes = [
        str(code)
        for code in city_codes
    ]

    target_codes = [
        str(code)
        for code in target_codes
    ]

    n = len(city_codes)

    if tuple(
        predicted_od.shape
    ) != (n, n):
        raise ValueError(
            f"predicted_od shape 오류: "
            f"{tuple(predicted_od.shape)}"
        )

    if tuple(
        actual_od.shape
    ) != (n, n):
        raise ValueError(
            f"actual_od shape 오류: "
            f"{tuple(actual_od.shape)}"
        )

    code_to_idx = {
        code: i
        for i, code
        in enumerate(city_codes)
    }

    target_set = set(
        target_codes
    )

    # 신도시 가상동 제외
    existing_indices = [
        i
        for i, code
        in enumerate(city_codes)
        if code not in target_set
    ]

    existing_codes = [
        city_codes[i]
        for i in existing_indices
    ]

    # ========================================================
    # 기존 행정동 실제 OD 벡터
    # ========================================================

    actual_patterns = []

    for idx in existing_indices:

        outgoing = (
            actual_od[
                idx,
                existing_indices,
            ]
        )

        incoming = (
            actual_od[
                existing_indices,
                idx,
            ]
        )

        actual_patterns.append(
            torch.cat(
                [
                    outgoing,
                    incoming,
                ]
            )
        )

    actual_patterns = (
        torch.stack(
            actual_patterns
        )
        .float()
    )

    normalized_actual = (
        F.normalize(
            actual_patterns,
            p=2,
            dim=1,
        )
    )

    result = {}

    # ========================================================
    # 신도시 각 동
    # ========================================================

    for target_code in target_codes:

        if target_code not in code_to_idx:
            result[target_code] = []
            continue

        target_idx = (
            code_to_idx[
                target_code
            ]
        )

        target_out = (
            predicted_od[
                target_idx,
                existing_indices,
            ]
            .float()
        )

        target_in = (
            predicted_od[
                existing_indices,
                target_idx,
            ]
            .float()
        )

        target_pattern = torch.cat(
            [
                target_out,
                target_in,
            ]
        )

        if (
            torch.linalg.vector_norm(
                target_pattern
            ).item()
            == 0
        ):
            result[target_code] = []
            continue

        normalized_target = (
            F.normalize(
                target_pattern.unsqueeze(
                    0
                ),
                p=2,
                dim=1,
            )
        )

        similarity = (
            normalized_target
            @ normalized_actual.T
        ).squeeze(0)

        k = min(
            top_k,
            len(existing_codes),
        )

        top_scores, top_indices = (
            torch.topk(
                similarity,
                k=k,
            )
        )

        items = []

        for (
            overall_score,
            relative_idx,
        ) in zip(
            top_scores.tolist(),
            top_indices.tolist(),
        ):

            candidate_code = (
                existing_codes[
                    relative_idx
                ]
            )

            candidate_idx = (
                existing_indices[
                    relative_idx
                ]
            )

            candidate_out = (
                actual_od[
                    candidate_idx,
                    existing_indices,
                ]
                .float()
            )

            candidate_in = (
                actual_od[
                    existing_indices,
                    candidate_idx,
                ]
                .float()
            )

            outflow_similarity = (
                _cosine(
                    target_out,
                    candidate_out,
                )
            )

            inflow_similarity = (
                _cosine(
                    target_in,
                    candidate_in,
                )
            )

            common_outflow = (
                _get_common_patterns(
                    target_out,
                    candidate_out,
                    existing_codes,
                    candidate_code=
                        candidate_code,
                    top_k=
                        reason_top_k,
                )
            )

            common_inflow = (
                _get_common_patterns(
                    target_in,
                    candidate_in,
                    existing_codes,
                    candidate_code=
                        candidate_code,
                    top_k=
                        reason_top_k,
                )
            )

            items.append(
                {
                    # 프론트 클릭 시
                    # 지도 이동에 바로 사용
                    "dong_code":
                        candidate_code,

                    "similarity":
                        float(
                            overall_score
                        ),

                    "outflow_similarity":
                        outflow_similarity,

                    "inflow_similarity":
                        inflow_similarity,

                    # 비교동 실제 총량도 웹에서
                    # 신도시와 비교 가능하도록 반환
                    "candidate_total_outflow":
                        float(
                            candidate_out.sum().item()
                        ),

                    "candidate_total_inflow":
                        float(
                            candidate_in.sum().item()
                        ),

                    "common_outflow_patterns":
                        common_outflow,

                    "common_inflow_patterns":
                        common_inflow,
                }
            )

        result[target_code] = items

    return result


# ============================================================
# 2023 실제 OD 로드
# ============================================================

def load_actual_od_2023(
    city_codes: list[str],
) -> torch.Tensor:

    path = (
        BASE_DIR
        / "newtown"
        / "od_data_2023.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"2023 OD 파일 없음:\n{path}"
        )

    print(
        "\n2023 실제 OD 로드 중:",
        path,
    )

    df = pd.read_csv(
        path
    )

    purposes = [
        "귀가",
        "출근",
        "등교",
        "업무",
        "기타",
    ]

    df["O_dong_code"] = (
        df["O_dong_code"]
        .astype(str)
        .str.replace(
            r"\.0$",
            "",
            regex=True,
        )
    )

    df["D_dong_code"] = (
        df["D_dong_code"]
        .astype(str)
        .str.replace(
            r"\.0$",
            "",
            regex=True,
        )
    )

    df["total_trips"] = (
        df[
            purposes
        ].sum(
            axis=1
        )
    )

    city_codes = [
        str(code)
        for code in city_codes
    ]

    code_to_idx = {
        code: i
        for i, code
        in enumerate(city_codes)
    }

    n = len(
        city_codes
    )

    actual_od = torch.zeros(
        (n, n),
        dtype=torch.float32,
    )

    for row in df.itertuples(
        index=False
    ):

        origin = str(
            row.O_dong_code
        )

        destination = str(
            row.D_dong_code
        )

        if (
            origin not in code_to_idx
            or
            destination
            not in code_to_idx
        ):
            continue

        actual_od[
            code_to_idx[
                origin
            ],
            code_to_idx[
                destination
            ],
        ] += float(
            row.total_trips
        )

    return actual_od


# ============================================================
# MAE 실행
# ============================================================

def run_mae_and_load_actual_od(
    *,
    newtown_name: str,
    period: str,
    device: str = "cpu",
):

    from ..newtown_predictor import (
        NewtownPreprocessor,
    )

    from ..predictor import (
        MAEPredictor,
    )

    print(
        "\nMAE 입력 데이터 준비 중..."
    )

    preprocessor = (
        NewtownPreprocessor(
            newtown_name=
                newtown_name,
            period=
                period,
        )
    )

    inputs = (
        preprocessor.prepare()
    )

    print(
        "MAE 모델 로드 중..."
    )

    predictor = (
        MAEPredictor(
            device=device
        )
    )

    print(
        "사용 체크포인트:",
        predictor.weight_path.name,
    )

    print(
        "MAE 예측 실행 중..."
    )

    (
        predicted_od,
        _metadata,
        origins,
        _destinations,
        zones,
    ) = predictor._run_full(
        inputs,
        request_metadata=None,
    )

    origins = [
        str(code)
        for code in origins
    ]

    zones = [
        str(code)
        for code in zones
    ]

    actual_od = (
        load_actual_od_2023(
            origins
        )
    )

    return (
        predicted_od.float(),
        actual_od.float(),
        origins,
        zones,
    )