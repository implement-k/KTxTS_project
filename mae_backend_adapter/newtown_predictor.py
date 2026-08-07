"""번들 신도시 데이터로 MAE 예측 입력을 만들고 결과를 반환한다."""

from __future__ import annotations

import json
import pickle
from collections.abc import Collection
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .predictor import (
    InputValidationError,
    MAEPredictor,
    ModelInputs,
    PreprocessingConfigurationError,
)
from .static_scaler import load_static_scaler


SUPPORTED_NEWTOWNS = ("all", "changneung", "gyosan", "wangsuk")
SUPPORTED_PERIODS = ("initial", "middle", "final")
DEFAULT_PERIOD = "final"
PURPOSE_COLUMNS = ("귀가", "출근", "등교", "업무", "기타")


def _validate_choice(name: str, value: str, supported: Collection[str]) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InputValidationError(f"{name}은 비어 있지 않은 문자열이어야 합니다.")
    clean = value.strip()
    if clean not in supported:
        raise InputValidationError(
            f"지원되지 않는 {name}입니다: {clean}. 지원 목록: {', '.join(supported)}"
        )
    return clean


def _normalize_code(value: Any) -> str:
    if pd.isna(value):
        raise PreprocessingConfigurationError("행정동 코드에 빈 값이 있습니다.")
    try:
        return str(int(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise PreprocessingConfigurationError(f"올바르지 않은 행정동 코드입니다: {value!r}") from exc


def _normalize_code_column(frame: pd.DataFrame, column: str) -> None:
    if column not in frame.columns:
        raise PreprocessingConfigurationError(f"필수 column이 없습니다: {column}")
    frame[column] = frame[column].map(_normalize_code)


class NewtownPreprocessor:
    """도시·시기별 번들 파일을 canonical ``ModelInputs``로 변환한다."""

    supported_newtowns = frozenset(SUPPORTED_NEWTOWNS)
    supported_periods = frozenset(SUPPORTED_PERIODS)

    def __init__(self, newtown_name: str, period: str = DEFAULT_PERIOD) -> None:
        self.newtown_name = _validate_choice(
            "newtown", newtown_name, SUPPORTED_NEWTOWNS
        )
        self.period = _validate_choice("period", period, SUPPORTED_PERIODS)
        self.package_root = Path(__file__).resolve().parent
        self.base_dir = self.package_root / "newtown" / self.newtown_name

    def prepare(self) -> ModelInputs:
        if not self.base_dir.is_dir():
            raise PreprocessingConfigurationError(
                f"신도시 데이터 폴더가 없습니다: {self.base_dir}"
            )

        codes = self._load_codes()
        x_dist = self._load_distance(codes)
        a_spatial = self._load_adjacency(codes)
        static_df, feature_names = self._load_static_features(codes)
        newtown_zone_codes = self._load_newtown_zone_codes(codes)
        zone_set = set(newtown_zone_codes)
        mask_tensor = torch.tensor(
            [code in zone_set for code in codes], dtype=torch.bool
        )
        x_od_masked = self._load_masked_od(codes, mask_tensor)

        scaler = load_static_scaler()
        scaled_features = scaler.transform_with_indicators(
            static_df[feature_names].to_numpy(),
            feature_names=feature_names,
            is_masked=mask_tensor.numpy(),
            is_merged=np.zeros(len(codes), dtype=bool),
        )

        return ModelInputs(
            x_static=torch.tensor(scaled_features, dtype=torch.float64),
            x_od_masked=x_od_masked,
            x_dist=x_dist,
            a_spatial=a_spatial,
            mask=mask_tensor,
            city_codes=codes,
            newtown_zone_codes=newtown_zone_codes,
            population_allocation_method="bundled_newtown_dataset",
            metadata={
                "newtown": self.newtown_name,
                "period": self.period,
                "year": "2023",
            },
        )

    def _load_codes(self) -> list[str]:
        frame = pd.read_excel(self.base_dir / "OD_dong_list_2023.xlsx")
        _normalize_code_column(frame, "dong_code")
        codes = frame["dong_code"].tolist()
        if not codes or len(codes) != len(set(codes)):
            raise PreprocessingConfigurationError(
                "OD_dong_list_2023.xlsx의 dong_code는 비어 있지 않고 중복이 없어야 합니다."
            )
        return codes

    def _load_distance(self, codes: list[str]) -> torch.Tensor:
        frame = pd.read_csv(
            self.base_dir / "dong_distance.csv",
            usecols=["O_dong_code", "D_dong_code", "distance"],
        )
        _normalize_code_column(frame, "O_dong_code")
        _normalize_code_column(frame, "D_dong_code")
        matrix = frame.pivot(
            index="O_dong_code", columns="D_dong_code", values="distance"
        ).reindex(index=codes, columns=codes)
        if matrix.isna().any().any():
            raise PreprocessingConfigurationError(
                f"{self.newtown_name} 거리 행렬에 누락된 OD 쌍이 있습니다."
            )
        values = matrix.to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or np.any(values < 0):
            raise PreprocessingConfigurationError("거리 값은 유한한 0 이상의 숫자여야 합니다.")
        return torch.log1p(torch.from_numpy(values))

    def _load_adjacency(self, codes: list[str]) -> torch.Tensor:
        with (self.base_dir / "dong_adjacency.pkl").open("rb") as file_obj:
            adjacency = pickle.load(file_obj)
        if not isinstance(adjacency, dict):
            raise PreprocessingConfigurationError("인접 데이터는 dict 형식이어야 합니다.")

        code_to_index = {code: index for index, code in enumerate(codes)}
        matrix = np.zeros((len(codes), len(codes)), dtype=np.float64)
        for node, neighbors in adjacency.items():
            node_index = code_to_index.get(_normalize_code(node))
            if node_index is None:
                continue
            for neighbor in neighbors:
                neighbor_index = code_to_index.get(_normalize_code(neighbor))
                if neighbor_index is None:
                    continue
                matrix[node_index, neighbor_index] = 1.0
                matrix[neighbor_index, node_index] = 1.0
        np.fill_diagonal(matrix, 0.0)
        return torch.from_numpy(matrix)

    def _load_static_features(self, codes: list[str]) -> tuple[pd.DataFrame, list[str]]:
        frame = pd.read_csv(self.base_dir / f"static_features_{self.period}.csv")
        _normalize_code_column(frame, "dong_code")
        if frame["dong_code"].duplicated().any():
            raise PreprocessingConfigurationError("static feature의 dong_code에 중복이 있습니다.")
        missing_codes = set(codes) - set(frame["dong_code"])
        if missing_codes:
            raise PreprocessingConfigurationError(
                f"static feature에 없는 dong_code입니다: {sorted(missing_codes)}"
            )
        frame = frame.set_index("dong_code").reindex(codes).reset_index()
        feature_names = sorted(
            column for column in frame.columns if column not in {"dong_code", "dong_name"}
        )
        if not feature_names:
            raise PreprocessingConfigurationError("static feature column이 없습니다.")
        frame[feature_names] = frame[feature_names].fillna(0)
        return frame, feature_names

    def _load_newtown_zone_codes(self, codes: list[str]) -> list[str]:
        mask_path = self.base_dir / "mask_code.json"
        try:
            mask_data = json.loads(mask_path.read_text(encoding="utf-8"))
            mask_city = mask_data["mask_city"]
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise PreprocessingConfigurationError(
                f"mask code 파일을 읽을 수 없습니다: {mask_path}"
            ) from exc
        if not isinstance(mask_city, dict) or not mask_city:
            raise PreprocessingConfigurationError("mask_city는 비어 있지 않은 object여야 합니다.")
        mask_codes = {_normalize_code(value) for value in mask_city.values()}
        missing = mask_codes - set(codes)
        if missing:
            raise PreprocessingConfigurationError(
                f"도시 node 목록에 없는 mask code입니다: {sorted(missing)}"
            )
        return [code for code in codes if code in mask_codes]

    def _load_masked_od(
        self, codes: list[str], mask_tensor: torch.Tensor
    ) -> torch.Tensor:
        frame = pd.read_csv(
            self.package_root / "newtown" / "od_data_2023.csv",
            usecols=["O_dong_code", "D_dong_code", *PURPOSE_COLUMNS],
        )
        _normalize_code_column(frame, "O_dong_code")
        _normalize_code_column(frame, "D_dong_code")
        valid = frame["O_dong_code"].isin(codes) & frame["D_dong_code"].isin(codes)
        selected = frame.loc[valid, ["O_dong_code", "D_dong_code", *PURPOSE_COLUMNS]].copy()
        selected["total"] = selected[list(PURPOSE_COLUMNS)].sum(axis=1)
        grouped = selected.groupby(
            ["O_dong_code", "D_dong_code"], sort=False
        )["total"].sum()
        matrix = grouped.unstack(fill_value=0).reindex(
            index=codes, columns=codes, fill_value=0
        )
        values = matrix.to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or np.any(values < 0):
            raise PreprocessingConfigurationError("OD 값은 유한한 0 이상의 숫자여야 합니다.")
        tensor = torch.log1p(torch.from_numpy(values))
        masked_pairs = mask_tensor.unsqueeze(1) | mask_tensor.unsqueeze(0)
        return torch.where(masked_pairs, torch.zeros_like(tensor), tensor)


class NewtownPredictor:
    """모델을 한 번 로드하고 도시 코드만 바꿔 반복 예측한다."""

    def __init__(
        self,
        device: str = "cpu",
        *,
        predictor: MAEPredictor | None = None,
    ) -> None:
        self.predictor = predictor or MAEPredictor(device=device)

    def predict(
        self, newtown_name: str, period: str = DEFAULT_PERIOD
    ) -> dict[str, Any]:
        preprocessor = NewtownPreprocessor(newtown_name, period)
        inputs = preprocessor.prepare()
        return self.predictor.predict_from_tensors(
            inputs,
            request_metadata={
                "newtown": preprocessor.newtown_name,
                "period": preprocessor.period,
            },
        )


def predict_newtown(
    newtown_name: str,
    period: str = DEFAULT_PERIOD,
    *,
    device: str = "cpu",
) -> dict[str, Any]:
    """단발성 호출용 편의 함수. 서버에서는 ``NewtownPredictor``를 재사용한다."""

    return NewtownPredictor(device=device).predict(newtown_name, period)
