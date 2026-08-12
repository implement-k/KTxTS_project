"""mae-year 백엔드 입력·출력 계약과 내부 모델 smoke test."""

from __future__ import annotations

import json
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import torch
from torch import nn

from mae_backend_adapter import (
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

    def forward(self, x_static, x_od_masked, x_dist, a_spatial, mask):
        del x_static, x_dist, a_spatial
        node_count = x_od_masked.shape[-1]
        predicted = torch.arange(
            1,
            node_count * node_count + 1,
            dtype=x_od_masked.dtype,
            device=x_od_masked.device,
        ).reshape(1, node_count, node_count)
        masked_pairs = mask.unsqueeze(-1) | mask.unsqueeze(-2)
        return torch.where(masked_pairs, predicted, x_od_masked) + (
            self.feature_embed[0].weight.mean() * 0
        )


class NegativeTinyODModel(TinyODModel):
    def forward(self, x_static, x_od_masked, x_dist, a_spatial, mask):
        del x_static, x_dist, a_spatial, mask
        return torch.full_like(x_od_masked, -1.0) + self.feature_embed[0].weight.mean() * 0


class TinyRunner:
    def __init__(self, model: nn.Module | None = None) -> None:
        self.model = (model or TinyODModel()).eval()
        self.num_features = 2
        self.forward_calls = 0

    def forward(self, tensors):
        self.forward_calls += 1
        with torch.inference_mode():
            return self.model(*tensors)


class TestPreprocessor:
    supported_newtowns = ("gyosan",)

    def __init__(self, *, bad_adjacency_shape: bool = False) -> None:
        self.bad_adjacency_shape = bad_adjacency_shape

    def prepare(self, *, newtown, total_population, age_ratios):
        del newtown, total_population, age_ratios
        return ModelInputs(
            x_static=torch.zeros(3, 2, dtype=torch.float64),
            x_od_masked=torch.tensor(
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 9.0]]
            ),
            x_dist=torch.zeros(3, 3),
            a_spatial=(
                torch.zeros(3, 4) if self.bad_adjacency_shape else torch.zeros(3, 3)
            ),
            mask=torch.tensor([True, True, False]),
            origin_codes=("ZONE_A", "ZONE_B", "EXISTING"),
            destination_codes=("ZONE_A", "ZONE_B", "EXISTING"),
            newtown_zone_codes=("ZONE_A", "ZONE_B"),
            population_allocation_method="test_fixture",
            output_transform="identity",
        )


def make_predictor(
    *,
    model: nn.Module | None = None,
    preprocessor: TestPreprocessor | None = None,
) -> tuple[MAEPredictor, TinyRunner, mock.Mock]:
    runner = TinyRunner(model)
    with mock.patch(
        "mae_backend_adapter.predictor._TorchMAERunner", return_value=runner
    ) as runner_class:
        predictor = MAEProvider(
            device="cpu",
            preprocessor=preprocessor,
        )
    return predictor, runner, runner_class


class PredictorContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.predictor, self.runner, self.runner_class = make_predictor(
            preprocessor=TestPreprocessor()
        )

    def test_constructor_loads_once_and_sets_eval(self) -> None:
        self.runner_class.assert_called_once()
        self.assertFalse(self.predictor.model.training)
        self.assertIsNone(self.predictor.num_nodes)
        self.assertEqual(self.predictor.num_features, 2)

    def test_same_adapter_accepts_two_dynamic_node_counts(self) -> None:
        def make_inputs(node_count: int) -> ModelInputs:
            codes = tuple(f"NODE_{index}" for index in range(node_count))
            x_od_masked = torch.ones(node_count, node_count)
            x_od_masked[0, :] = 0
            x_od_masked[:, 0] = 0
            return ModelInputs(
                x_static=torch.zeros(node_count, 2),
                x_od_masked=x_od_masked,
                x_dist=torch.zeros(node_count, node_count),
                a_spatial=torch.zeros(node_count, node_count),
                mask=torch.tensor([True] + [False] * (node_count - 1)),
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

    def test_missing_internal_checkpoint_is_explicit(self) -> None:
        with self.assertRaisesRegex(CheckpointLoadError, "파일이 없습니다"):
            MAEPredictor(weight_file_name="missing.pth")

    def test_public_request_validation(self) -> None:
        for population in (-1, 3.5, True, "100000"):
            with self.subTest(population=population), self.assertRaises(InputValidationError):
                self.predictor.predict(
                    newtown="gyosan", total_population=population, age_ratios=VALID_RATIOS
                )
        with self.assertRaises(InputValidationError):
            self.predictor.predict(
                newtown="unknown", total_population=100000, age_ratios=VALID_RATIOS
            )
        with self.assertRaises(InputValidationError):
            self.predictor.predict(
                newtown="gyosan",
                total_population=100000,
                age_ratios={**VALID_RATIOS, "0_19": 0.21},
            )

    def test_missing_preprocessor_is_explicit(self) -> None:
        predictor, _, _ = make_predictor()
        with self.assertRaisesRegex(PreprocessingConfigurationError, "전처리기"):
            predictor.predict(
                newtown="gyosan", total_population=100000, age_ratios=VALID_RATIOS
            )

    def test_filtered_od_json_and_repeat_without_reload(self) -> None:
        first = self.predictor.predict(
            newtown="gyosan", total_population=100000, age_ratios=VALID_RATIOS
        )
        second = self.predictor.predict(
            newtown="gyosan", total_population=100000, age_ratios=VALID_RATIOS
        )
        self.assertEqual(self.runner.forward_calls, 2)
        self.assertEqual(first, second)
        self.assertEqual(first["newtown"], "gyosan")
        self.assertEqual(first["newtown_zone_codes"], ["ZONE_A", "ZONE_B"])
        self.assertNotIn("od_matrix", first)
        self.assertEqual(len(first["od"]), 8)
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
        self.assertNotIn("self_loop_policy", first["metadata"])
        json.dumps(first, ensure_ascii=False, allow_nan=False)

    def test_negative_model_outputs_are_clamped_and_excluded(self) -> None:
        predictor, _, _ = make_predictor(model=NegativeTinyODModel())
        inputs = TestPreprocessor().prepare(
            newtown="gyosan", total_population=100000, age_ratios=VALID_RATIOS
        )
        result = predictor.predict_from_tensors(inputs)
        self.assertEqual(result["od"], [])
        self.assertEqual(result["metadata"]["negative_values_policy"], "clamped_to_zero")

    def test_five_tensor_shape_dtype_and_preprocessing_contract(self) -> None:
        base = TestPreprocessor().prepare(
            newtown="gyosan", total_population=100000, age_ratios=VALID_RATIOS
        )
        with self.assertRaisesRegex(TensorShapeError, "a_spatial"):
            self.predictor.predict_from_tensors(
                TestPreprocessor(bad_adjacency_shape=True).prepare(
                    newtown="gyosan", total_population=100000, age_ratios=VALID_RATIOS
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

    def test_masked_od_rows_and_columns_must_be_zero(self) -> None:
        base = TestPreprocessor().prepare(
            newtown="gyosan", total_population=100000, age_ratios=VALID_RATIOS
        )
        self.assertTrue(self.predictor.predict_from_tensors(base)["od"])

        row_value = base.x_od_masked.clone()
        row_value[0, 2] = 1.0
        with self.assertRaisesRegex(
            PreprocessingConfigurationError, "mask=True.*행과 열"
        ):
            self.predictor.predict_from_tensors(
                replace(base, x_od_masked=row_value)
            )

        column_value = base.x_od_masked.clone()
        column_value[2, 0] = 1.0
        with self.assertRaisesRegex(
            PreprocessingConfigurationError, "mask=True.*행과 열"
        ):
            self.predictor.predict_from_tensors(
                replace(base, x_od_masked=column_value)
            )

    def test_duplicate_codes_are_aggregated_and_missing_code_becomes_unmapped(self) -> None:
        inputs = TestPreprocessor().prepare(
            newtown="gyosan", total_population=100000, age_ratios=VALID_RATIOS
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
        self.assertEqual(by_pair[("ZONE", "UNMAPPED")], 9.0)
        self.assertEqual(by_pair[("UNMAPPED", "ZONE")], 15.0)
        self.assertNotIn(("UNMAPPED", "UNMAPPED"), by_pair)


class OutputAdapterTests(unittest.TestCase):
    def test_duplicate_od_is_summed_and_external_to_external_is_excluded(self) -> None:
        rows = ODOutputAdapter().adapt(
            torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
            origin_codes=("ZONE", "ZONE", "UNMAPPED"),
            destination_codes=("ZONE", "ZONE", "UNMAPPED"),
            newtown_zone_codes=("ZONE",),
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
    def test_internal_checkpoint_cpu_forward(self) -> None:
        root = Path(__file__).resolve().parents[1]
        if not (root / "model" / "mae.pth").is_file():
            self.skipTest("실제 checkpoint가 없습니다.")

        predictor = MAEProvider(device="cpu")
        node_count = 4
        codes = tuple(f"NODE_{index:04d}" for index in range(node_count))
        inputs = ModelInputs(
            x_static=torch.zeros(node_count, 20),
            x_od_masked=torch.zeros(node_count, node_count),
            x_dist=torch.zeros(node_count, node_count),
            a_spatial=torch.zeros(node_count, node_count),
            mask=torch.tensor([True, False, False, False]),
            origin_codes=codes,
            destination_codes=codes,
            newtown_zone_codes=(codes[0],),
            population_allocation_method="repository_model_smoke",
            output_transform="log1p",
            metadata={"year": "2023"},
        )
        result = predictor.predict_from_tensors(
            inputs, request_metadata={"newtown": "smoke"}
        )

        self.assertFalse(predictor.model.training)
        self.assertIsNone(predictor.num_nodes)
        self.assertEqual(predictor.num_features, 20)
        self.assertEqual(result["metadata"]["node_count"], node_count)
        self.assertEqual(result["metadata"]["feature_count"], 20)
        self.assertEqual(result["metadata"]["checkpoint"], "mae.pth")
        self.assertNotIn("self_loop_policy", result["metadata"])
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
