"""MAE 백엔드 공개 계약, OD 필터 및 실제 v7 smoke test."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import torch
from torch import nn

from mae_backend_adapter import (
    CheckpointCompatibilityError,
    CheckpointLoadError,
    InputValidationError,
    MAEPredictor,
    ModelInputs,
    PreprocessingConfigurationError,
    TensorShapeError,
)


VALID_RATIOS = {
    "0_19": 0.20,
    "20_39": 0.30,
    "40_64": 0.35,
    "65_plus": 0.15,
}


class TinyODModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature_embed = nn.Sequential(nn.Linear(2, 2))

    def forward(self, x_static, x_od_masked, x_dist, mask):
        del x_static, x_dist, mask
        return x_od_masked + self.feature_embed[0].weight.mean()


class TestPreprocessor:
    supported_newtowns = ("교산",)

    def __init__(self, *, bad_shape: bool = False) -> None:
        self.bad_shape = bad_shape

    def prepare(self, *, newtown, total_population, age_ratios):
        del newtown, total_population, age_ratios
        return ModelInputs(
            x_static=torch.zeros(2, 2),
            x_od_masked=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            x_dist=torch.zeros(2, 3) if self.bad_shape else torch.zeros(2, 2),
            mask=torch.tensor([True, False]),
            origin_codes=("GYOSAN_01", "OLD_01"),
            destination_codes=("GYOSAN_01", "OLD_01"),
            newtown_zone_codes=("GYOSAN_01",),
            population_allocation_method="official_test_fixture",
            output_transform="identity",
        )


class PredictorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.checkpoint = Path(self.temp_dir.name) / "tiny.pth"
        torch.save(TinyODModel().state_dict(), self.checkpoint)
        self.factory_calls = 0

        def factory(state_dict, model_config):
            del state_dict, model_config
            self.factory_calls += 1
            return TinyODModel()

        self.factory = factory
        self.predictor = MAEPredictor(
            self.checkpoint,
            preprocessor=TestPreprocessor(),
            model_factory=self.factory,
            use_lgbm_self_loop=False,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_constructor_loads_once_and_sets_eval(self) -> None:
        self.assertEqual(self.factory_calls, 1)
        self.assertFalse(self.predictor.model.training)

    def test_missing_checkpoint_and_lfs_pointer(self) -> None:
        with self.assertRaisesRegex(CheckpointLoadError, "파일이 없습니다"):
            MAEPredictor(
                Path(self.temp_dir.name) / "missing.pth",
                model_factory=self.factory,
                use_lgbm_self_loop=False,
            )
        pointer = Path(self.temp_dir.name) / "pointer.pth"
        pointer.write_text("version https://git-lfs.github.com/spec/v1\n", encoding="utf-8")
        with self.assertRaisesRegex(CheckpointLoadError, "Git LFS pointer"):
            MAEPredictor(pointer, model_factory=self.factory, use_lgbm_self_loop=False)

    def test_input_validation(self) -> None:
        for population in (-1, 3.5, True, "100000"):
            with self.subTest(population=population), self.assertRaises(InputValidationError):
                self.predictor.predict(
                    newtown="교산", total_population=population, age_ratios=VALID_RATIOS
                )
        with self.assertRaises(InputValidationError):
            self.predictor.predict(
                newtown="왕숙", total_population=100000, age_ratios=VALID_RATIOS
            )
        with self.assertRaises(InputValidationError):
            self.predictor.predict(
                newtown="교산",
                total_population=100000,
                age_ratios={**VALID_RATIOS, "0_19": 0.21},
            )

    def test_missing_preprocessor_is_explicit(self) -> None:
        predictor = MAEPredictor(
            self.checkpoint,
            supported_newtowns=("교산",),
            model_factory=self.factory,
            use_lgbm_self_loop=False,
        )
        with self.assertRaisesRegex(PreprocessingConfigurationError, "전처리기"):
            predictor.predict(
                newtown="교산", total_population=100000, age_ratios=VALID_RATIOS
            )

    def test_filtered_od_movement_types_json_and_repeat_without_reload(self) -> None:
        with mock.patch("mae_backend_adapter.predictor.torch.load", wraps=torch.load) as load_spy:
            predictor = MAEPredictor(
                self.checkpoint,
                preprocessor=TestPreprocessor(),
                model_factory=self.factory,
                use_lgbm_self_loop=False,
            )
            first = predictor.predict(
                newtown="교산", total_population=100000, age_ratios=VALID_RATIOS
            )
            second = predictor.predict(
                newtown="교산", total_population=100000, age_ratios=VALID_RATIOS
            )
        self.assertEqual(load_spy.call_count, 1)
        self.assertEqual(first, second)
        self.assertNotIn("od_matrix", first)
        self.assertEqual(first["newtown_zone_codes"], ["GYOSAN_01"])
        self.assertEqual(
            [row["movement_type"] for row in first["od"]],
            ["internal", "outflow", "inflow"],
        )
        self.assertEqual(first["metadata"]["node_count"], 2)
        json.dumps(first, ensure_ascii=False, allow_nan=False)

    def test_explicit_full_matrix_method(self) -> None:
        inputs = TestPreprocessor().prepare(
            newtown="교산", total_population=100000, age_ratios=VALID_RATIOS
        )
        result = self.predictor.predict_full_matrix_from_tensors(inputs)
        self.assertEqual(len(result["od_matrix"]), 2)
        self.assertNotIn("od", result)

    def test_lightgbm_input_is_float32_for_every_execution_path(self) -> None:
        received_dtypes = []

        def self_loop_predictor(x_static):
            received_dtypes.append(x_static.dtype)
            return torch.zeros(x_static.shape[0])

        predictor = MAEPredictor(
            self.checkpoint,
            model_factory=self.factory,
            self_loop_predictor=self_loop_predictor,
        )
        inputs = TestPreprocessor().prepare(
            newtown="교산", total_population=100000, age_ratios=VALID_RATIOS
        )
        inputs = replace(inputs, x_static=inputs.x_static.to(torch.float64))
        predictor.predict_from_tensors(inputs)
        self.assertEqual(received_dtypes, [torch.float32])

    def test_tensor_and_zone_configuration_errors(self) -> None:
        with self.assertRaisesRegex(TensorShapeError, "x_dist"):
            self.predictor.predict_from_tensors(
                TestPreprocessor(bad_shape=True).prepare(
                    newtown="교산", total_population=100000, age_ratios=VALID_RATIOS
                )
            )
        missing_zone = ModelInputs(
            x_static=torch.zeros(2, 2),
            x_od_masked=torch.zeros(2, 2),
            x_dist=torch.zeros(2, 2),
            mask=torch.zeros(2, dtype=torch.bool),
            origin_codes=("A", "B"),
            destination_codes=("A", "B"),
            newtown_zone_codes=("C",),
            population_allocation_method="fixture",
            output_transform="identity",
        )
        with self.assertRaisesRegex(PreprocessingConfigurationError, "없는 신도시 zone"):
            self.predictor.predict_from_tensors(missing_zone)


class RepositoryV7SmokeTests(unittest.TestCase):
    def test_model_import_error_points_to_required_file_and_import_path(self) -> None:
        root = Path(__file__).resolve().parents[2]
        original_import = __import__

        def fail_model_import(name, *args, **kwargs):
            if name == "src.mae.models":
                raise ModuleNotFoundError(name)
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=fail_model_import):
            with self.assertRaisesRegex(
                CheckpointCompatibilityError, "Python import path.*src/mae/models.py"
            ):
                MAEPredictor(
                    root / "best_model" / "mae:v7.pth",
                    use_lgbm_self_loop=False,
                )

    def test_actual_v7_strict_cpu_forward_and_lgbm_override(self) -> None:
        root = Path(__file__).resolve().parents[2]
        predictor = MAEPredictor(
            root / "best_model" / "mae:v7.pth",
            device="cpu",
            supported_newtowns=("교산",),
        )
        self.assertEqual(predictor.num_nodes, 1137)
        self.assertEqual(predictor.num_features, 15)
        inputs = ModelInputs(
            x_static=torch.zeros(1137, 15),
            x_od_masked=torch.zeros(1137, 1137),
            x_dist=torch.zeros(1137, 1137),
            mask=torch.tensor([True] + [False] * 1136),
            origin_codes=tuple(f"NODE_{i:04d}" for i in range(1137)),
            destination_codes=tuple(f"NODE_{i:04d}" for i in range(1137)),
            newtown_zone_codes=("NODE_0000",),
            population_allocation_method="synthetic_tensor_smoke_only",
            output_transform="log1p",
        )
        result = predictor.predict_from_tensors(
            inputs, request_metadata={"newtown": "교산"}
        )
        repeated = predictor.predict_from_tensors(
            inputs, request_metadata={"newtown": "교산"}
        )
        self.assertFalse(predictor.model.training)
        self.assertEqual(result, repeated)
        self.assertEqual(len(result["od"]), 2273)
        self.assertEqual(result["metadata"]["self_loop_policy"], "lightgbm_override")
        self.assertEqual(
            result["metadata"]["lightgbm_execution"],
            "worker" if sys.platform == "darwin" else "in_process",
        )
        json.dumps(result, allow_nan=False)

    def test_v7_rejects_variable_node_count_before_forward(self) -> None:
        root = Path(__file__).resolve().parents[2]
        predictor = MAEPredictor(
            root / "best_model" / "mae:v7.pth",
            supported_newtowns=("교산",),
            use_lgbm_self_loop=False,
        )
        inputs = ModelInputs(
            x_static=torch.zeros(4, 15),
            x_od_masked=torch.zeros(4, 4),
            x_dist=torch.zeros(4, 4),
            mask=torch.zeros(4, dtype=torch.bool),
            origin_codes=("A", "B", "C", "D"),
            destination_codes=("A", "B", "C", "D"),
            newtown_zone_codes=("A",),
            population_allocation_method="synthetic",
        )
        with self.assertRaisesRegex(TensorShapeError, "고정 1137"):
            predictor.predict_from_tensors(inputs)


if __name__ == "__main__":
    unittest.main()
