from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..static_scaler import load_static_scaler


BASE_DIR = Path(__file__).resolve().parents[1]


def get_newtown_dir(newtown_name: str) -> Path:
    path = BASE_DIR / "newtown" / newtown_name

    if not path.exists():
        raise FileNotFoundError(
            f"신도시 데이터 폴더가 없습니다:\n{path}"
        )

    return path


def load_city_codes(
    newtown_name: str,
) -> list[str]:

    path = (
        get_newtown_dir(newtown_name)
        / "OD_dong_list_2023.xlsx"
    )

    df = pd.read_excel(path)

    return (
        df["dong_code"]
        .astype(str)
        .str.replace(
            r"\.0$",
            "",
            regex=True,
        )
        .tolist()
    )


def load_target_codes(
    newtown_name: str,
) -> list[str]:

    path = (
        get_newtown_dir(newtown_name)
        / "mask_code.json"
    )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    mask_city = data.get(
        "mask_city",
        {},
    )

    return [
        str(code)
        for code
        in mask_city.values()
    ]


def load_adjacency(
    newtown_name: str,
    city_codes: list[str],
) -> torch.Tensor:

    path = (
        get_newtown_dir(newtown_name)
        / "dong_adjacency.pkl"
    )

    with open(
        path,
        "rb",
    ) as f:
        adjacency_dict = pickle.load(f)

    n = len(city_codes)

    code_to_idx = {
        str(code): idx
        for idx, code
        in enumerate(city_codes)
    }

    matrix = np.zeros(
        (n, n),
        dtype=np.float32,
    )

    for node, neighbors in (
        adjacency_dict.items()
    ):

        node = str(node)

        if node not in code_to_idx:
            continue

        i = code_to_idx[node]

        for neighbor in neighbors:

            neighbor = str(neighbor)

            if neighbor not in code_to_idx:
                continue

            j = code_to_idx[neighbor]

            matrix[i, j] = 1.0
            matrix[j, i] = 1.0

    return torch.tensor(
        matrix,
        dtype=torch.float32,
    )


def load_static_features(
    newtown_name: str,
    period: str,
    city_codes: list[str],
):
    """
    반환:
    - x_static_scaled : (N, 18)
    - x_static_raw    : (N, 18)
    - code_to_name
    - feature_names
    """

    path = (
        get_newtown_dir(newtown_name)
        / f"static_features_{period}.csv"
    )

    static_df = pd.read_csv(
        path,
        dtype={"dong_code": str},
    )

    static_df["dong_code"] = (
        static_df["dong_code"]
        .astype(str)
        .str.replace(
            r"\.0$",
            "",
            regex=True,
        )
    )

    static_df = (
        static_df
        .set_index("dong_code")
        .reindex(city_codes)
        .reset_index()
    )

    if static_df["dong_name"].isna().any():

        missing = static_df.loc[
            static_df["dong_name"].isna(),
            "dong_code",
        ].tolist()

        raise ValueError(
            f"Static Feature 누락 코드: {missing[:20]}"
        )

    raw_feature_names = [
        col
        for col in static_df.columns
        if col not in {
            "dong_code",
            "dong_name",
        }
    ]

    scaler = load_static_scaler()

    # 학습 당시 feature 순서로 정렬
    ordered_raw = static_df[
        list(scaler.feature_names)
].to_numpy(
        dtype=np.float64
    )

    scaled = scaler.transform(
        static_df[
            raw_feature_names
        ].values,
        feature_names=raw_feature_names,
    )

    x_static_scaled = torch.tensor(
        scaled,
        dtype=torch.float32,
    )

    x_static_raw = torch.tensor(
        ordered_raw,
        dtype=torch.float32,
    )

    code_to_name = dict(
        zip(
            static_df["dong_code"].astype(str),
            static_df["dong_name"].astype(str),
        )
    )

    return (
        x_static_scaled,
        x_static_raw,
        code_to_name,
        scaler.feature_names,
    )