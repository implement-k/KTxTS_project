"""mae-year 백엔드 입력·출력 계약과 실제 최종 checkpoint smoke test."""

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
    MAEProvider,
    ModelInputs,
    ODOutputAdapter,
    PreprocessingConfigurationError,
    TensorShapeError,
)


VALID_RATIOS = {
    "0_19": 0.20,
    "20_59": 0.60,
    "60_plus": 0.20,
}


class TinyODModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature_embed = nn.Sequential(nn.Linear(2, 2))
        nn.init.zeros_(self.feature_embed[0].weight)
        nn.init.zeros_(self.feature_embed[0].bias)

    def forward(
        self,
        x_static,
        x_od_masked,
        x_dist,
        a_spatial,
        mask,
        active_node_mask,
    ):
        del x_static, x_dist, a_spatial, mask, active_node_mask
        return x_od_masked + self.feature_embed[0].weight.mean() * 0


class NegativeTinyODModel(TinyODModel):
    def forward(
        self,
        x_static,
        x_od_masked,
        x_dist,
        a_spatial,
        mask,
        active_node_mask,
    ):
        del x_static, x_dist, a_spatial, mask, active_node_mask
        return torch.full_like(x_od_masked, -1.0) + self.feature_embed[0].weight.mean() * 0


class TestPreprocessor:
    supported_newtowns = ("교산",)

    def __init__(self, *, bad_adjacency_shape: bool = False) -> None:
        self.bad_adjacency_shape = bad_adjacency_shape

    def prepare(self, *, newtown, total_population, age_ratios):
        del newtown, total_population, age_ratios
        return ModelInputs(
            x_static=torch.zeros(3, 2, dtype=torch.float64),
            x_od_masked=torch.tensor(
                [[1.0, 2.0, 0.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
            ),
            x_dist=torch.zeros(3, 3),
            a_spatial=(
                torch.zeros(3, 4) if self.bad_adjacency_shape else torch.zeros(3, 3)
            ),
            mask=torch.tensor([True, True, False]),
            active_node_mask=torch.ones(3, dtype=torch.bool),
            origin_codes=("ZONE_A", "ZONE_B", "EXISTING"),
            destination_codes=("ZONE_A", "ZONE_B", "EXISTING"),
            newtown_zone_codes=("ZONE_A", "ZONE_B"),
            population_allocation_method="test_fixture",
            output_transform="identity",
        )


class PredictorContractTests(unittest.TestCase):
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
        self.predictor = MAEProvider(
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
        self.assertIsNone(self.predictor.num_nodes)
        self.assertEqual(self.predictor.num_features, 2)

    def test_same_adapter_accepts_two_dynamic_node_counts(self) -> None:
        def make_inputs(node_count: int) -> ModelInputs:
            codes = tuple(f"NODE_{index}" for index in range(node_count))
            return ModelInputs(
                x_static=torch.zeros(node_count, 2),
                x_od_masked=torch.ones(node_count, node_count),
                x_dist=torch.zeros(node_count, node_count),
                a_spatial=torch.zeros(node_count, node_count),
                mask=torch.tensor([True] + [False] * (node_count - 1)),
                active_node_mask=torch.ones(node_count, dtype=torch.bool),
                origin_codes=codes,
                destination_codes=codes,
                newtown_zone_codes=(codes[0],),
                population_allocation_method="dynamic_node_test",
                output_transform="identity",
            )

        for node_count in (2, 5):
            with self.subTest(node_count=node_count):
                result = self.predictor.predict_from_tensors(make_inputs(node_count))
                self.assertEqual(result["metadata"]["node_count"], node_count)
                self.assertEqual(len(result["od"]), node_count * 2 - 1)

    def test_missing_checkpoint_lfs_pointer_and_strict_mismatch(self) -> None:
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

        def incompatible_factory(state_dict, model_config):
            del state_dict, model_config
            return nn.Linear(3, 1)

        with self.assertRaises(CheckpointCompatibilityError):
            MAEPredictor(
                self.checkpoint,
                model_factory=incompatible_factory,
                use_lgbm_self_loop=False,
            )

    def test_public_request_validation(self) -> None:
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

    def test_filtered_od_json_zero_exclusion_and_repeat_without_reload(self) -> None:
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
        self.assertEqual(first["newtown"], "교산")
        self.assertEqual(first["newtown_zone_codes"], ["ZONE_A", "ZONE_B"])
        self.assertNotIn("od_matrix", first)
        self.assertEqual(len(first["od"]), 7)
        self.assertTrue(all(row["predicted_trips"] > 0 for row in first["od"]))
        movements = {
            (row["origin_code"], row["destination_code"]): row["movement_type"]
            for row in first["od"]
        }
        self.assertNotIn(("EXISTING", "EXISTING"), movements)
        self.assertEqual(movements[("ZONE_A", "ZONE_B")], "internal")
        self.assertEqual(movements[("ZONE_B", "EXISTING")], "outflow")
        self.assertEqual(movements[("EXISTING", "ZONE_A")], "inflow")
        self.assertEqual(first["metadata"]["zero_od_policy"], "excluded")
        json.dumps(first, ensure_ascii=False, allow_nan=False)

    def test_negative_model_outputs_are_clamped_and_excluded(self) -> None:
        def negative_factory(state_dict, model_config):
            del state_dict, model_config
            return NegativeTinyODModel()

        predictor = MAEPredictor(
            self.checkpoint,
            model_factory=negative_factory,
            use_lgbm_self_loop=False,
        )
        inputs = TestPreprocessor().prepare(
            newtown="교산", total_population=100000, age_ratios=VALID_RATIOS
        )
        result = predictor.predict_from_tensors(inputs)
        self.assertEqual(result["od"], [])
        self.assertEqual(result["metadata"]["negative_values_policy"], "clamped_to_zero")

    def test_lightgbm_gets_float32_and_overrides_diagonal(self) -> None:
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
        result = predictor.predict_from_tensors(inputs)
        self.assertEqual(received_dtypes, [torch.float32])
        self.assertFalse(
            any(row["origin_code"] == row["destination_code"] for row in result["od"])
        )
        self.assertEqual(result["metadata"]["self_loop_policy"], "lightgbm_override")

    def test_six_tensor_shape_dtype_and_preprocessing_contract(self) -> None:
        base = TestPreprocessor().prepare(
            newtown="교산", total_population=100000, age_ratios=VALID_RATIOS
        )
        with self.assertRaisesRegex(TensorShapeError, "a_spatial"):
            self.predictor.predict_from_tensors(
                TestPreprocessor(bad_adjacency_shape=True).prepare(
                    newtown="교산", total_population=100000, age_ratios=VALID_RATIOS
                )
            )
        with self.assertRaisesRegex(TensorShapeError, "mask는"):
            self.predictor.predict_from_tensors(
                replace(base, mask=torch.tensor([True, True]))
            )
        with self.assertRaisesRegex(TensorShapeError, "code 수"):
            self.predictor.predict_from_tensors(
                replace(base, origin_codes=("ZONE_A", "ZONE_B"))
            )
        with self.assertRaisesRegex(TensorShapeError, "dtype"):
            self.predictor.predict_from_tensors(
                replace(base, mask=base.mask.to(torch.int64))
            )
        with self.assertRaisesRegex(PreprocessingConfigurationError, "mask=True"):
            self.predictor.predict_from_tensors(
                replace(base, mask=torch.tensor([True, False, False]))
            )
        with self.assertRaisesRegex(TensorShapeError, "0 이상"):
            self.predictor.predict_from_tensors(
                replace(base, x_dist=torch.full((3, 3), -1.0))
            )

    def test_inactive_nodes_are_not_returned(self) -> None:
        inputs = TestPreprocessor().prepare(
            newtown="교산", total_population=100000, age_ratios=VALID_RATIOS
        )
        x_od = inputs.x_od_masked.clone()
        x_od[2, :] = 0
        x_od[:, 2] = 0
        inputs = replace(
            inputs,
            x_od_masked=x_od,
            active_node_mask=torch.tensor([True, True, False]),
        )
        result = self.predictor.predict_from_tensors(inputs)
        self.assertEqual(len(result["od"]), 4)
        self.assertFalse(
            any("EXISTING" in (row["origin_code"], row["destination_code"]) for row in result["od"])
        )

    def test_duplicate_codes_are_aggregated_and_missing_code_becomes_unmapped(self) -> None:
        inputs = TestPreprocessor().prepare(
            newtown="교산", total_population=100000, age_ratios=VALID_RATIOS
        )
        inputs = replace(
            inputs,
            origin_codes=("ZONE", "ZONE", None),
            destination_codes=("ZONE", "ZONE", None),
            newtown_zone_codes=("ZONE",),
        )
        result = self.predictor.predict_from_tensors(inputs)
        by_pair = {
            (row["origin_code"], row["destination_code"]): row["predicted_trips"]
            for row in result["od"]
        }
        self.assertEqual(by_pair[("ZONE", "ZONE")], 12.0)
        self.assertEqual(by_pair[("ZONE", "UNMAPPED")], 6.0)
        self.assertEqual(by_pair[("UNMAPPED", "ZONE")], 15.0)
        self.assertNotIn(("UNMAPPED", "UNMAPPED"), by_pair)


class OutputAdapterTests(unittest.TestCase):
    def test_duplicate_od_is_summed_and_external_to_external_is_excluded(self) -> None:
        rows = ODOutputAdapter().adapt(
            torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
            origin_codes=("ZONE", "ZONE", "UNMAPPED"),
            destination_codes=("ZONE", "ZONE", "UNMAPPED"),
            newtown_zone_codes=("ZONE",),
            active_node_mask=(True, True, True),
        )
        by_pair = {
            (row["origin_code"], row["destination_code"]): row["predicted_trips"]
            for row in rows
        }
        self.assertEqual(by_pair[("ZONE", "ZONE")], 12.0)
        self.assertEqual(by_pair[("ZONE", "UNMAPPED")], 9.0)
        self.assertEqual(by_pair[("UNMAPPED", "ZONE")], 15.0)
        self.assertNotIn(("UNMAPPED", "UNMAPPED"), by_pair)


class RepositoryFinalModelSmokeTests(unittest.TestCase):
    def test_actual_checkpoint_dataset_2023_cpu_forward_and_lgbm(self) -> None:
        root = Path(__file__).resolve().parents[2]
        checkpoint = root / "best_model" / "mae:hybrid-86epoch.pth"
        if not checkpoint.is_file():
            self.skipTest("실제 checkpoint가 없습니다.")

        mae_source = str(root / "src" / "mae-year")
        src_source = str(root / "src")
        for import_path in (src_source, mae_source):
            if import_path not in sys.path:
                sys.path.insert(0, import_path)
        from dataset import ODDataset

        dataset = ODDataset(year="2023", use_merge_train=False)
        sample = dataset[0]
        node_count = dataset.num_nodes
        zone_index = int(torch.nonzero(sample["mask"], as_tuple=False)[0])
        codes = tuple(f"NODE_{index:04d}" for index in range(node_count))
        predictor = MAEProvider(
            checkpoint,
            device="cpu",
            supported_newtowns=("smoke",),
        )
        inputs = ModelInputs(
            x_static=sample["X_static"],
            x_od_masked=sample["X_OD_masked"],
            x_dist=sample["X_dist"],
            a_spatial=sample["A_spatial"],
            mask=sample["mask"],
            active_node_mask=sample["active_node_mask"],
            origin_codes=codes,
            destination_codes=codes,
            newtown_zone_codes=(codes[zone_index],),
            population_allocation_method="repository_dataset_smoke",
            output_transform="log1p",
            metadata={"year": "2023"},
        )
        result = predictor.predict_from_tensors(
            inputs, request_metadata={"newtown": "smoke"}
        )
        self.assertFalse(predictor.model.training)
        self.assertIsNone(predictor.num_nodes)
        self.assertEqual(predictor.num_features, 20)
        self.assertEqual(result["metadata"]["node_count"], dataset.num_nodes)
        self.assertEqual(result["metadata"]["feature_count"], 20)
        self.assertEqual(result["metadata"]["model_family"], "mae-year/ODMAE")
        self.assertEqual(result["metadata"]["self_loop_policy"], "lightgbm_override")
        self.assertEqual(
            result["metadata"]["lightgbm_execution"],
            "worker" if sys.platform == "darwin" else "in_process",
        )
        self.assertTrue(result["od"])
        self.assertTrue(all(row["predicted_trips"] > 0 for row in result["od"]))
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
