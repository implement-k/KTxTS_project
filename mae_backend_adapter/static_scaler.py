"""2023 mae-year static feature scaler artifact를 안전하게 로드하고 재사용한다."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .predictor import PreprocessingConfigurationError


ARTIFACT_DIRECTORY = Path(__file__).resolve().parent / "artifacts"
DEFAULT_SCALER_NPZ_PATH = ARTIFACT_DIRECTORY / "mae_year_2023_static_scaler.npz"
DEFAULT_SCALER_JSON_PATH = ARTIFACT_DIRECTORY / "mae_year_2023_static_scaler.json"
INDICATOR_FEATURE_NAMES = ("is_masked", "is_merged")
_EXPECTED_MASKING_FEATURE_NAMES = frozenset(
    {
        "worker_count",
        "business_count",
        "worker_density",
        "business_density",
    }
)
_SCHEMA_VERSION = 1
_NPZ_KEYS = frozenset(
    {
        "mean_",
        "scale_",
        "var_",
        "n_samples_seen_",
        "dong_codes",
        "train_indices",
        "val_indices",
        "test_indices",
    }
)


class StaticScalerArtifactError(PreprocessingConfigurationError):
    """Scaler artifact나 raw static feature 계약이 맞지 않을 때 발생한다."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_read_only_float64(value: np.ndarray, *, name: str, feature_count: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (feature_count,):
        raise StaticScalerArtifactError(
            f"scaler {name} shape은 ({feature_count},)여야 하지만 {array.shape}입니다."
        )
    if not np.isfinite(array).all():
        raise StaticScalerArtifactError(f"scaler {name}에 NaN 또는 무한대가 있습니다.")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class StaticFeatureScaler:
    """Feature schema와 StandardScaler parameter를 함께 보유하는 불변 런타임 객체."""

    year: str
    feature_names: tuple[str, ...]
    masking_feature_names: tuple[str, ...]
    masking_feature_indices: tuple[int, ...]
    mean_: np.ndarray
    scale_: np.ndarray
    var_: np.ndarray
    metadata: Mapping[str, Any]

    @property
    def feature_count(self) -> int:
        return len(self.feature_names)

    def transform(
        self,
        raw_values: Any,
        *,
        feature_names: Sequence[str],
    ) -> np.ndarray:
        """Raw feature를 artifact schema 순서로 재정렬한 뒤 변환한다."""

        incoming_names = tuple(feature_names)
        invalid_names = [name for name in incoming_names if not isinstance(name, str) or not name]
        if invalid_names:
            raise StaticScalerArtifactError(
                "feature_names는 비어 있지 않은 문자열만 포함해야 합니다."
            )
        duplicate_names = sorted(
            {name for name in incoming_names if incoming_names.count(name) > 1}
        )
        if duplicate_names:
            raise StaticScalerArtifactError(f"중복 static feature입니다: {duplicate_names}")

        expected = set(self.feature_names)
        incoming = set(incoming_names)
        missing = sorted(expected - incoming)
        unexpected = sorted(incoming - expected)
        if missing or unexpected:
            raise StaticScalerArtifactError(
                f"static feature schema가 맞지 않습니다. 누락={missing}, 추가={unexpected}"
            )

        try:
            values = np.asarray(raw_values, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise StaticScalerArtifactError("raw static feature는 숫자여야 합니다.") from exc
        if values.ndim not in (1, 2):
            raise StaticScalerArtifactError(
                f"raw static feature는 1차원 또는 2차원이어야 하지만 {values.ndim}차원입니다."
            )
        if values.shape[-1] != len(incoming_names):
            raise StaticScalerArtifactError(
                f"raw static feature 마지막 차원은 feature_names 수 "
                f"{len(incoming_names)}와 같아야 하지만 {values.shape[-1]}입니다."
            )
        if not np.isfinite(values).all():
            raise StaticScalerArtifactError("raw static feature에 NaN 또는 무한대가 있습니다.")

        incoming_index = {name: index for index, name in enumerate(incoming_names)}
        reorder = [incoming_index[name] for name in self.feature_names]
        ordered = values[..., reorder]
        return (ordered - self.mean_) / self.scale_

    def transform_mapping(self, raw_features: Mapping[str, Any]) -> np.ndarray:
        """Raw feature mapping 한 행을 artifact schema로 정렬해 변환한다."""

        if not isinstance(raw_features, Mapping):
            raise StaticScalerArtifactError("raw_features는 feature name과 값의 mapping이어야 합니다.")
        names = tuple(raw_features.keys())
        return self.transform([raw_features[name] for name in names], feature_names=names)

    def transform_with_indicators(
        self,
        raw_values: Any,
        *,
        feature_names: Sequence[str],
        is_masked: Any,
        is_merged: Any,
    ) -> np.ndarray:
        """18개 raw feature를 scaling·masking한 후 indicator 2개를 붙인다."""

        scaled = self.transform(raw_values, feature_names=feature_names).copy()
        leading_shape = scaled.shape[:-1]
        masked = self._indicator_array(is_masked, leading_shape, name="is_masked")
        merged = self._indicator_array(is_merged, leading_shape, name="is_merged")

        if scaled.ndim == 1:
            if bool(masked):
                scaled[list(self.masking_feature_indices)] = 0.0
        else:
            masked_rows = np.flatnonzero(masked == 1.0)
            if masked_rows.size:
                scaled[np.ix_(masked_rows, self.masking_feature_indices)] = 0.0

        indicators = np.stack((masked, merged), axis=-1)
        return np.concatenate((scaled, indicators), axis=-1)

    @staticmethod
    def _indicator_array(value: Any, shape: tuple[int, ...], *, name: str) -> np.ndarray:
        try:
            array = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise StaticScalerArtifactError(f"{name}은 0 또는 1이어야 합니다.") from exc
        if shape == ():
            if array.ndim != 0:
                raise StaticScalerArtifactError(f"단일 feature 행의 {name}은 scalar여야 합니다.")
        elif array.ndim == 0:
            array = np.full(shape, float(array), dtype=np.float64)
        elif array.shape != shape:
            raise StaticScalerArtifactError(
                f"{name} shape은 {shape}여야 하지만 {array.shape}입니다."
            )
        if not np.isfinite(array).all() or not np.isin(array, (0.0, 1.0)).all():
            raise StaticScalerArtifactError(f"{name}은 0 또는 1만 포함해야 합니다.")
        return array


def _load_static_scaler(npz_path: Path, json_path: Path) -> StaticFeatureScaler:
    if not npz_path.is_file():
        raise StaticScalerArtifactError(f"scaler NPZ 파일이 없습니다: {npz_path}")
    if not json_path.is_file():
        raise StaticScalerArtifactError(f"scaler JSON 파일이 없습니다: {json_path}")
    try:
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StaticScalerArtifactError(f"scaler JSON을 읽을 수 없습니다: {json_path}: {exc}") from exc

    if metadata.get("schema_version") != _SCHEMA_VERSION:
        raise StaticScalerArtifactError(
            f"지원하지 않는 scaler schema_version입니다: {metadata.get('schema_version')!r}"
        )
    year = metadata.get("year")
    feature_names_value = metadata.get("feature_names")
    if not isinstance(feature_names_value, list) or not all(
        isinstance(name, str) and name for name in feature_names_value
    ):
        raise StaticScalerArtifactError("JSON feature_names는 비어 있지 않은 문자열 배열이어야 합니다.")
    feature_names = tuple(feature_names_value)
    if len(feature_names) != len(set(feature_names)):
        raise StaticScalerArtifactError("JSON feature_names에 중복이 있습니다.")
    if len(feature_names) != 18:
        raise StaticScalerArtifactError(
            f"2023 raw static feature는 18개여야 하지만 {len(feature_names)}개입니다."
        )
    if metadata.get("feature_count") != len(feature_names):
        raise StaticScalerArtifactError("JSON feature_count와 feature_names 수가 다릅니다.")
    if tuple(metadata.get("indicator_feature_names", ())) != INDICATOR_FEATURE_NAMES:
        raise StaticScalerArtifactError(
            f"indicator feature 순서는 {INDICATOR_FEATURE_NAMES}여야 합니다."
        )

    masking_feature_names_value = metadata.get("masking_feature_names")
    if not isinstance(masking_feature_names_value, list) or not all(
        isinstance(name, str) and name for name in masking_feature_names_value
    ):
        raise StaticScalerArtifactError(
            "JSON masking_feature_names는 비어 있지 않은 문자열 배열이어야 합니다."
        )
    masking_feature_names = tuple(masking_feature_names_value)
    if len(masking_feature_names) != len(set(masking_feature_names)):
        raise StaticScalerArtifactError("JSON masking_feature_names에 중복이 있습니다.")
    masking_names = set(masking_feature_names)
    missing_masking_names = sorted(_EXPECTED_MASKING_FEATURE_NAMES - masking_names)
    unexpected_masking_names = sorted(masking_names - _EXPECTED_MASKING_FEATURE_NAMES)
    if missing_masking_names or unexpected_masking_names:
        raise StaticScalerArtifactError(
            "JSON masking feature가 현재 Dataset 계약과 다릅니다. "
            f"누락={missing_masking_names}, 추가={unexpected_masking_names}"
        )

    masking_feature_indices_value = metadata.get("masking_feature_indices")
    if not isinstance(masking_feature_indices_value, list) or not all(
        isinstance(index, int) and not isinstance(index, bool)
        for index in masking_feature_indices_value
    ):
        raise StaticScalerArtifactError("JSON masking_feature_indices는 정수 배열이어야 합니다.")
    masking_feature_indices = tuple(masking_feature_indices_value)
    if len(masking_feature_indices) != len(masking_feature_names):
        raise StaticScalerArtifactError(
            "JSON masking_feature_indices 수와 masking_feature_names 수가 다릅니다."
        )
    if len(masking_feature_indices) != len(set(masking_feature_indices)):
        raise StaticScalerArtifactError("JSON masking_feature_indices에 중복이 있습니다.")
    invalid_masking_indices = [
        index
        for index in masking_feature_indices
        if index < 0 or index >= len(feature_names)
    ]
    if invalid_masking_indices:
        raise StaticScalerArtifactError(
            f"JSON masking_feature_indices가 feature 범위를 벗어납니다: {invalid_masking_indices}"
        )
    mismatched_masking_indices = [
        (name, index)
        for name, index in zip(masking_feature_names, masking_feature_indices)
        if feature_names[index] != name
    ]
    if mismatched_masking_indices:
        raise StaticScalerArtifactError(
            "JSON masking feature 이름과 index가 일치하지 않습니다: "
            f"{mismatched_masking_indices}"
        )

    scaler_metadata = metadata.get("scaler")
    if not isinstance(scaler_metadata, Mapping):
        raise StaticScalerArtifactError("JSON scaler metadata가 mapping이 아닙니다.")
    expected_npz_hash = scaler_metadata.get("npz_sha256")
    actual_npz_hash = _sha256(npz_path)
    if expected_npz_hash != actual_npz_hash:
        raise StaticScalerArtifactError(
            f"scaler NPZ SHA-256가 JSON과 다릅니다: expected={expected_npz_hash}, actual={actual_npz_hash}"
        )

    try:
        with np.load(npz_path, allow_pickle=False) as archive:
            keys = frozenset(archive.files)
            if keys != _NPZ_KEYS:
                raise StaticScalerArtifactError(
                    f"scaler NPZ key가 맞지 않습니다. 누락={sorted(_NPZ_KEYS - keys)}, "
                    f"추가={sorted(keys - _NPZ_KEYS)}"
                )
            mean = _as_read_only_float64(
                archive["mean_"], name="mean_", feature_count=len(feature_names)
            )
            scale = _as_read_only_float64(
                archive["scale_"], name="scale_", feature_count=len(feature_names)
            )
            variance = _as_read_only_float64(
                archive["var_"], name="var_", feature_count=len(feature_names)
            )
            dong_codes = np.asarray(archive["dong_codes"], dtype=np.int64)
            train_indices = np.asarray(archive["train_indices"], dtype=np.int64)
            val_indices = np.asarray(archive["val_indices"], dtype=np.int64)
            test_indices = np.asarray(archive["test_indices"], dtype=np.int64)
            n_samples_seen = int(np.asarray(archive["n_samples_seen_"]).item())
    except StaticScalerArtifactError:
        raise
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise StaticScalerArtifactError(f"scaler NPZ를 읽을 수 없습니다: {npz_path}: {exc}") from exc

    if np.any(scale <= 0):
        raise StaticScalerArtifactError("scaler scale_은 모두 0보다 커야 합니다.")
    if np.any(variance < 0):
        raise StaticScalerArtifactError("scaler var_는 모두 0 이상이어야 합니다.")
    expected_scale = np.where(variance == 0.0, 1.0, np.sqrt(variance))
    if not np.allclose(scale, expected_scale, rtol=1e-12, atol=0.0):
        raise StaticScalerArtifactError("scaler scale_과 var_가 StandardScaler 규칙에 맞지 않습니다.")

    split = metadata.get("split")
    if not isinstance(split, Mapping):
        raise StaticScalerArtifactError("JSON split metadata가 mapping이 아닙니다.")
    expected_counts = {
        "all_count": len(dong_codes),
        "train_count": len(train_indices),
        "val_count": len(val_indices),
        "test_count": len(test_indices),
    }
    for key, actual_count in expected_counts.items():
        if split.get(key) != actual_count:
            raise StaticScalerArtifactError(
                f"JSON split.{key}와 NPZ 배열 크기가 다릅니다: "
                f"json={split.get(key)!r}, npz={actual_count}"
            )
    if n_samples_seen != len(train_indices):
        raise StaticScalerArtifactError(
            f"n_samples_seen_={n_samples_seen}와 train 크기={len(train_indices)}가 다릅니다."
        )
    if scaler_metadata.get("n_samples_seen") != n_samples_seen:
        raise StaticScalerArtifactError(
            "JSON scaler.n_samples_seen과 NPZ n_samples_seen_가 다릅니다."
        )
    all_split_indices = np.concatenate((train_indices, val_indices, test_indices))
    if len(np.unique(all_split_indices)) != len(dong_codes) or set(
        all_split_indices.tolist()
    ) != set(range(len(dong_codes))):
        raise StaticScalerArtifactError("train/val/test indices가 전체 node를 중복 없이 분할하지 않습니다.")

    return StaticFeatureScaler(
        year=year,
        feature_names=feature_names,
        masking_feature_names=masking_feature_names,
        masking_feature_indices=masking_feature_indices,
        mean_=mean,
        scale_=scale,
        var_=variance,
        metadata=metadata,
    )


@lru_cache(maxsize=None)
def _load_static_scaler_cached(npz_path: str, json_path: str) -> StaticFeatureScaler:
    return _load_static_scaler(Path(npz_path), Path(json_path))


def load_static_scaler(
    npz_path: str | Path = DEFAULT_SCALER_NPZ_PATH,
    json_path: str | Path = DEFAULT_SCALER_JSON_PATH,
) -> StaticFeatureScaler:
    """Artifact를 절대 경로로 정규화해 process 당 한 번만 로드한다."""

    resolved_npz = str(Path(npz_path).expanduser().resolve())
    resolved_json = str(Path(json_path).expanduser().resolve())
    return _load_static_scaler_cached(resolved_npz, resolved_json)


def clear_static_scaler_cache() -> None:
    """테스트와 명시적 artifact 교체 시에만 사용한다."""

    _load_static_scaler_cached.cache_clear()
