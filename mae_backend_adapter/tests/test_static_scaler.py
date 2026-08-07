"""2023 static scaler artifact의 Dataset 동등성과 백엔드 입력 계약을 검증한다."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from mae_backend_adapter import (
    INDICATOR_FEATURE_NAMES,
    StaticScalerArtifactError,
    clear_static_scaler_cache,
    load_static_scaler,
)


class StaticScalerContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[2]
        cls.scaler = load_static_scaler()

    @classmethod
    def tearDownClass(cls) -> None:
        clear_static_scaler_cache()

    def test_artifact_load_is_cached_and_has_expected_contract(self) -> None:
        self.assertIs(self.scaler, load_static_scaler())
        self.assertEqual(self.scaler.year, "2023")
        self.assertEqual(self.scaler.feature_count, 18)
        self.assertEqual(INDICATOR_FEATURE_NAMES, ("is_masked", "is_merged"))
        self.assertEqual(
            self.scaler.masking_feature_names,
            ("worker_count", "business_count", "worker_density", "business_density"),
        )
        self.assertEqual(self.scaler.masking_feature_indices, (10, 0, 11, 1))
        self.assertEqual(
            self.scaler.metadata["checkpoint"]["sha256"],
            "4ba6cd4d24d5f85af5405ecc36aab6bbfbb8ff2462d8630fdec69d4079cd72cc",
        )
        checkpoint = self.root / self.scaler.metadata["checkpoint"]["path"]
        self.assertEqual(checkpoint.stat().st_size, 3885946)
        self.assertEqual(
            hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            self.scaler.metadata["checkpoint"]["sha256"],
        )

    def test_transform_with_indicators_reorders_and_masks_only_masked_rows(self) -> None:
        names = tuple(reversed(self.scaler.feature_names))
        target_scaled = np.vstack(
            (
                np.arange(1, self.scaler.feature_count + 1, dtype=np.float64),
                -np.arange(1, self.scaler.feature_count + 1, dtype=np.float64),
            )
        )
        canonical_raw = self.scaler.mean_ + target_scaled * self.scaler.scale_
        raw = canonical_raw[
            :, [self.scaler.feature_names.index(name) for name in names]
        ]
        original = raw.copy()

        transformed = self.scaler.transform(raw, feature_names=names)
        with_indicators = self.scaler.transform_with_indicators(
            raw,
            feature_names=names,
            is_masked=np.array([1, 0]),
            is_merged=np.array([0, 1]),
        )

        np.testing.assert_allclose(transformed, target_scaled, rtol=0.0, atol=1e-14)
        np.testing.assert_array_equal(raw, original)
        expected_with_mask = transformed.copy()
        expected_with_mask[0, list(self.scaler.masking_feature_indices)] = 0.0
        np.testing.assert_array_equal(with_indicators[:, :18], expected_with_mask)
        np.testing.assert_array_equal(with_indicators[:, 18:], [[1.0, 0.0], [0.0, 1.0]])
        changed_coordinates = set(zip(*np.nonzero(with_indicators[:, :18] != transformed)))
        self.assertEqual(
            changed_coordinates,
            {(0, index) for index in self.scaler.masking_feature_indices},
        )

    def test_transform_with_indicators_supports_single_row_without_mutating_input(self) -> None:
        raw = self.scaler.mean_ + self.scaler.scale_
        original = raw.copy()

        transformed = self.scaler.transform_with_indicators(
            raw,
            feature_names=self.scaler.feature_names,
            is_masked=1,
            is_merged=1,
        )

        expected = self.scaler.transform(raw, feature_names=self.scaler.feature_names)
        np.testing.assert_allclose(expected, np.ones(self.scaler.feature_count), atol=1e-15)
        expected[list(self.scaler.masking_feature_indices)] = 0.0
        np.testing.assert_array_equal(transformed, np.concatenate((expected, [1.0, 1.0])))
        np.testing.assert_array_equal(raw, original)

    def test_transform_mapping_uses_artifact_feature_order(self) -> None:
        raw = {
            name: float(index + 1)
            for index, name in enumerate(reversed(self.scaler.feature_names))
        }
        transformed = self.scaler.transform_mapping(raw)
        expected_raw = np.asarray([raw[name] for name in self.scaler.feature_names])
        np.testing.assert_array_equal(
            transformed,
            (expected_raw - self.scaler.mean_) / self.scaler.scale_,
        )

    def test_missing_extra_duplicate_dimension_and_nonfinite_features_fail(self) -> None:
        names = self.scaler.feature_names
        values = np.zeros(18)
        with self.assertRaisesRegex(StaticScalerArtifactError, "누락=.*추가="):
            self.scaler.transform(
                values,
                feature_names=(*names[1:], "unexpected_feature"),
            )
        with self.assertRaisesRegex(StaticScalerArtifactError, "중복 static feature"):
            self.scaler.transform(
                values,
                feature_names=(*names[:-1], names[0]),
            )
        with self.assertRaisesRegex(StaticScalerArtifactError, "마지막 차원"):
            self.scaler.transform(values[:-1], feature_names=names)
        nonfinite = values.copy()
        nonfinite[0] = np.nan
        with self.assertRaisesRegex(StaticScalerArtifactError, "NaN"):
            self.scaler.transform(nonfinite, feature_names=names)
        with self.assertRaisesRegex(StaticScalerArtifactError, "is_masked은 0 또는 1"):
            self.scaler.transform_with_indicators(
                values,
                feature_names=names,
                is_masked=2,
                is_merged=0,
            )

    def test_npz_checksum_mismatch_fails_without_fallback(self) -> None:
        source_npz = self.root / "mae_backend_adapter" / "artifacts" / "mae_year_2023_static_scaler.npz"
        source_json = self.root / "mae_backend_adapter" / "artifacts" / "mae_year_2023_static_scaler.json"
        with tempfile.TemporaryDirectory() as directory:
            temp_root = Path(directory)
            temp_npz = temp_root / source_npz.name
            temp_json = temp_root / source_json.name
            temp_npz.write_bytes(source_npz.read_bytes())
            metadata = json.loads(source_json.read_text(encoding="utf-8"))
            metadata["scaler"]["npz_sha256"] = "0" * 64
            temp_json.write_text(
                json.dumps(metadata, ensure_ascii=False),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(StaticScalerArtifactError, "SHA-256"):
                load_static_scaler(temp_npz, temp_json)

    def test_invalid_masking_metadata_fails_without_fallback(self) -> None:
        source_npz = self.root / "mae_backend_adapter" / "artifacts" / "mae_year_2023_static_scaler.npz"
        source_json = self.root / "mae_backend_adapter" / "artifacts" / "mae_year_2023_static_scaler.json"

        def assert_invalid(mutator, message_pattern: str) -> None:
            with tempfile.TemporaryDirectory() as directory:
                temp_root = Path(directory)
                temp_npz = temp_root / source_npz.name
                temp_json = temp_root / source_json.name
                temp_npz.write_bytes(source_npz.read_bytes())
                metadata = json.loads(source_json.read_text(encoding="utf-8"))
                mutator(metadata)
                temp_json.write_text(
                    json.dumps(metadata, ensure_ascii=False),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(StaticScalerArtifactError, message_pattern):
                    load_static_scaler(temp_npz, temp_json)

        cases = (
            (
                "missing names",
                lambda metadata: metadata.pop("masking_feature_names"),
                "masking_feature_names",
            ),
            (
                "duplicate names",
                lambda metadata: metadata.__setitem__(
                    "masking_feature_names", ["worker_count"] * 4
                ),
                "masking_feature_names에 중복",
            ),
            (
                "missing indices",
                lambda metadata: metadata.pop("masking_feature_indices"),
                "masking_feature_indices",
            ),
            (
                "duplicate indices",
                lambda metadata: metadata.__setitem__(
                    "masking_feature_indices", [10, 10, 11, 1]
                ),
                "masking_feature_indices에 중복",
            ),
            (
                "mismatched index",
                lambda metadata: metadata.__setitem__(
                    "masking_feature_indices", [2, 0, 11, 1]
                ),
                "이름과 index가 일치하지 않습니다",
            ),
        )
        for label, mutator, message_pattern in cases:
            with self.subTest(label=label):
                assert_invalid(mutator, message_pattern)


class CurrentDatasetScalerEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[2]
        src_path = str(cls.root / "src")
        mae_path = str(cls.root / "src" / "mae-year")
        for import_path in (src_path, mae_path):
            if import_path not in sys.path:
                sys.path.insert(0, import_path)

        dataset_path = cls.root / "src" / "mae-year" / "dataset.py"
        spec = importlib.util.spec_from_file_location("_scaler_test_dataset", dataset_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Dataset module을 로드할 수 없습니다: {dataset_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.dataset = module.ODDataset(year="2023", use_merge_train=False)
        cls.scaler = load_static_scaler()

        dong_df = pd.read_excel(cls.root / "dataset" / "raw" / "dong" / "OD_dong_list_2023.xlsx")
        dongs = dong_df["dong_code"].astype(int).to_numpy()
        static_df = pd.read_csv(cls.root / "dataset" / "final_static_features_2023.csv")
        static_df["dong_code"] = static_df["dong_code"].astype(int)
        cls.static_df = static_df.set_index("dong_code").reindex(dongs).reset_index()
        cls.static_df.fillna(0, inplace=True)

    def test_saved_parameters_and_split_arrays_equal_current_dataset(self) -> None:
        np.testing.assert_array_equal(self.scaler.mean_, self.dataset.scaler.mean_)
        np.testing.assert_array_equal(self.scaler.scale_, self.dataset.scaler.scale_)
        np.testing.assert_array_equal(self.scaler.var_, self.dataset.scaler.var_)

        npz_path = self.root / "mae_backend_adapter" / "artifacts" / "mae_year_2023_static_scaler.npz"
        with np.load(npz_path, allow_pickle=False) as artifact:
            np.testing.assert_array_equal(artifact["train_indices"], self.dataset.train_indices)
            np.testing.assert_array_equal(artifact["val_indices"], self.dataset.val_indices)
            np.testing.assert_array_equal(artifact["test_indices"], self.dataset.test_indices)

    def test_transform_with_indicators_equals_all_current_dataset_x_static_columns(self) -> None:
        raw_static = self.static_df[list(self.scaler.feature_names)].to_numpy()
        transformed = self.scaler.transform_with_indicators(
            raw_static,
            feature_names=self.scaler.feature_names,
            is_masked=self.dataset.X_static[:, -2],
            is_merged=self.dataset.X_static[:, -1],
        )

        self.assertEqual(transformed.shape, self.dataset.X_static.shape)
        np.testing.assert_allclose(
            transformed,
            self.dataset.X_static,
            rtol=0.0,
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
