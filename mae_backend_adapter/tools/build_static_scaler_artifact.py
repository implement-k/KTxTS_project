"""2023 ODDataset과 동일한 train-only StandardScaler artifact를 생성한다."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


YEAR = "2023"
NPZ_FILENAME = "mae_year_2023_static_scaler.npz"
JSON_FILENAME = "mae_year_2023_static_scaler.json"
INDICATOR_FEATURE_NAMES = ["is_masked", "is_merged"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_int_sequence(values: np.ndarray) -> str:
    canonical = ",".join(str(int(value)) for value in values).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def relative_file_metadata(repository_root: Path, path: Path, *, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "path": path.resolve().relative_to(repository_root.resolve()).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"module을 로드할 수 없습니다: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git_value(repository_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def build_artifacts(repository_root: Path, output_directory: Path) -> tuple[Path, Path]:
    repository_root = repository_root.resolve()
    dataset_source = repository_root / "src" / "mae-year" / "dataset.py"
    config_source = repository_root / "src" / "config.py"
    checkpoint_path = repository_root / "best_model" / "mae:hybrid-86epoch.pth"

    src_path = str(repository_root / "src")
    mae_path = str(repository_root / "src" / "mae-year")
    for import_path in (src_path, mae_path):
        if import_path not in sys.path:
            sys.path.insert(0, import_path)

    config = load_module("_mae_scaler_config", config_source)
    dataset_module = load_module("_mae_scaler_dataset", dataset_source)
    dataset = dataset_module.ODDataset(year=YEAR, use_merge_train=False)

    dong_path = Path(config.DONG_CODE_23_PATH)
    static_path = Path(config.STATIC_DATA_23_PATH)
    od_path = Path(config.OD_DATA_23_PATH)
    distance_path = Path(config.DIST_DATA_23_PATH)
    merge_cache_path = repository_root / "src" / "mae-year" / "merge_cache_2023.pkl"

    dong_df = pd.read_excel(dong_path)
    dongs = dong_df["dong_code"].astype(int).to_numpy()
    static_df = pd.read_csv(static_path)
    static_df["dong_code"] = static_df["dong_code"].astype(int)
    static_df = static_df.set_index("dong_code").reindex(dongs).reset_index()
    static_df.fillna(0, inplace=True)
    feature_names = sorted(
        column for column in static_df.columns if column not in {"dong_code", "dong_name"}
    )
    raw_static = static_df[feature_names].to_numpy()

    if len(feature_names) != 18:
        raise RuntimeError(f"raw static feature는 18개여야 하지만 {len(feature_names)}개입니다.")
    if dataset.num_nodes != len(dongs) or not np.array_equal(
        dataset.all_indices, np.arange(len(dongs))
    ):
        raise RuntimeError("Dataset node 순서와 raw static node 순서가 다릅니다.")

    # base_data_2023.pt를 복사하지 않고, 현재 raw static과 split으로 새로 fit한다.
    scaler = StandardScaler()
    scaler.fit(raw_static[dataset.train_indices])
    for parameter_name in ("mean_", "scale_", "var_"):
        generated = np.asarray(getattr(scaler, parameter_name))
        current_dataset = np.asarray(getattr(dataset.scaler, parameter_name))
        if not np.array_equal(generated, current_dataset):
            raise RuntimeError(
                f"새로 fit한 scaler {parameter_name}과 ODDataset scaler가 다릅니다."
            )

    independently_transformed = (raw_static - scaler.mean_) / scaler.scale_
    dataset_transformed = np.asarray(dataset.X_static[:, : len(feature_names)]).copy()
    independently_transformed[
        np.ix_(dataset.test_indices, dataset.masking_indices)
    ] = 0.0
    max_abs_diff = float(np.max(np.abs(independently_transformed - dataset_transformed)))
    if not np.allclose(independently_transformed, dataset_transformed, rtol=0.0, atol=1e-12):
        raise RuntimeError(
            f"생성한 scaler와 ODDataset.X_static이 다릅니다: max_abs_diff={max_abs_diff}"
        )

    output_directory.mkdir(parents=True, exist_ok=True)
    npz_path = output_directory / NPZ_FILENAME
    json_path = output_directory / JSON_FILENAME
    npz_temp = output_directory / f".{NPZ_FILENAME}.tmp"
    json_temp = output_directory / f".{JSON_FILENAME}.tmp"
    with npz_temp.open("wb") as file_obj:
        np.savez_compressed(
            file_obj,
            mean_=np.asarray(scaler.mean_, dtype=np.float64),
            scale_=np.asarray(scaler.scale_, dtype=np.float64),
            var_=np.asarray(scaler.var_, dtype=np.float64),
            n_samples_seen_=np.asarray(scaler.n_samples_seen_, dtype=np.int64),
            dong_codes=np.asarray(dongs, dtype=np.int64),
            train_indices=np.asarray(dataset.train_indices, dtype=np.int64),
            val_indices=np.asarray(dataset.val_indices, dtype=np.int64),
            test_indices=np.asarray(dataset.test_indices, dtype=np.int64),
        )
    npz_temp.replace(npz_path)

    dong_to_index = {int(code): index for index, code in enumerate(dongs)}

    def effective_groups(groups: dict[str, list[str]]) -> dict[str, list[int]]:
        return {
            city: [int(code) for code in codes if int(code) in dong_to_index]
            for city, codes in groups.items()
        }

    metadata = {
        "schema_version": 1,
        "artifact_type": "sklearn_standard_scaler",
        "year": YEAR,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "masking_feature_names": list(config.MASKING_COLUMNS),
        "masking_feature_indices": [feature_names.index(name) for name in config.MASKING_COLUMNS],
        "indicator_feature_names": INDICATOR_FEATURE_NAMES,
        "indicator_contract": (
            "is_masked and is_merged are excluded from StandardScaler; append them in this order "
            "after transforming the 18 raw static features"
        ),
        "scaler": {
            "implementation": "sklearn.preprocessing.StandardScaler",
            "fit_scope": "raw_static[train_indices] only",
            "npz_file": NPZ_FILENAME,
            "npz_sha256": sha256(npz_path),
            "n_samples_seen": int(scaler.n_samples_seen_),
            "dtype": "float64",
        },
        "split": {
            "algorithm": "train = all node indices - union(validation indices, test indices)",
            "index_basis": "zero-based row order of dataset/raw/dong/OD_dong_list_2023.xlsx",
            "all_count": len(dongs),
            "train_count": len(dataset.train_indices),
            "val_count": len(dataset.val_indices),
            "test_count": len(dataset.test_indices),
            "dong_codes_sha256": hash_int_sequence(dongs),
            "train_indices_sha256": hash_int_sequence(dataset.train_indices),
            "val_indices_sha256": hash_int_sequence(dataset.val_indices),
            "test_indices_sha256": hash_int_sequence(dataset.test_indices),
            "train_dong_codes_sha256": hash_int_sequence(dongs[dataset.train_indices]),
            "validation_groups": effective_groups(config.VAL_CITIES_23_CODES),
            "test_groups": effective_groups(config.TEST_CITIES_23_CODES),
        },
        "inputs": [
            relative_file_metadata(repository_root, dong_path, role="canonical_node_order"),
            relative_file_metadata(repository_root, static_path, role="scaler_raw_static_features"),
            relative_file_metadata(repository_root, od_path, role="checkpoint_assumed_training_od"),
            relative_file_metadata(repository_root, distance_path, role="checkpoint_assumed_training_distance"),
            relative_file_metadata(repository_root, merge_cache_path, role="checkpoint_assumed_training_merge_cache"),
        ],
        "source": {
            "git_commit": git_value(repository_root, "rev-parse", "HEAD"),
            "git_branch": git_value(repository_root, "branch", "--show-current"),
            "files": [
                relative_file_metadata(repository_root, dataset_source, role="dataset_implementation"),
                relative_file_metadata(repository_root, config_source, role="split_and_path_configuration"),
            ],
        },
        "checkpoint": relative_file_metadata(
            repository_root, checkpoint_path, role="target_checkpoint"
        ),
        "verification": {
            "reference": "ODDataset(year='2023').X_static[:, :18] after Dataset test masking",
            "rtol": 0.0,
            "atol": 1e-12,
            "max_abs_diff": max_abs_diff,
        },
    }
    json_temp.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    json_temp.replace(json_path)
    return npz_path, json_path


def main() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    output_directory = repository_root / "mae_backend_adapter" / "artifacts"
    npz_path, json_path = build_artifacts(repository_root, output_directory)
    print(f"saved: {npz_path}")
    print(f"saved: {json_path}")


if __name__ == "__main__":
    main()
