"""MAE adapter의 5-tensor 입력·출력 계약과 번들 모델 smoke test."""

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


VALID_RATIOS = {"0_19": 0.20, "20_59": 0.60, "60_plus": 0.20}


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


class TinyRunner:
    def __init__(self) -> None:
        self.model = TinyODModel().eval()
        self.num_features = 2
        self.forward_calls = 0

    def forward(self, tensors):
        self.forward_calls += 1
        with torch.inference_mode():
            return self.model(*tensors)


class TestPreprocessor:
    supported_newtowns = ("gyosan",)

    def prepare(self, *, newtown, total_population, age_ratios):
        del newtown, total_population, age_ratios
        return make_inputs()


def make_inputs(node_count: int = 3) -> ModelInputs:
    codes = tuple(f"NODE_{index}" for index in range(node_count))
    x_od_masked = torch.ones(node_count, node_count)
    x_od_masked[0, :] = 0
    x_od_masked[:, 0] = 0
    return ModelInputs(
        x_static=torch.zeros(node_count, 2, dtype=torch.float64),
        x_od_masked=x_od_masked,
        x_dist=torch.zeros(node_count, node_count),
        a_spatial=torch.zeros(node_count, node_count),
        mask=torch.tensor([True] + [False] * (node_count - 1)),
        city_codes=codes,
        newtown_zone_codes=(codes[0],),
        population_allocation_method="test_fixture",
        output_transform="identity",
    )


def make_predictor(*, preprocessor=None):
    runner = TinyRunner()
    with mock.patch(
        "mae_backend_adapter.predictor._TorchMAERunner", return_value=runner
    ) as runner_class:
        predictor = MAEProvider(device="cpu", preprocessor=preprocessor)
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

    def test_same_predictor_accepts_dynamic_node_counts(self) -> None:
        for node_count in (2, 5):
            with self.subTest(node_count=node_count):
                result = self.predictor.predict_from_tensors(make_inputs(node_count))
                self.assertEqual(result["metadata"]["node_count"], node_count)
                self.assertEqual(len(result["od"]), node_count * 2 - 1)

    def test_public_request_validation_and_missing_preprocessor(self) -> None:
        for population in (-1, 3.5, True, "100000"):
            with self.subTest(population=population), self.assertRaises(InputValidationError):
                self.predictor.predict(
                    newtown="gyosan",
                    total_population=population,
                    age_ratios=VALID_RATIOS,
                )
        with self.assertRaises(InputValidationError):
            self.predictor.predict(
                newtown="unknown", total_population=100000, age_ratios=VALID_RATIOS
            )
        predictor, _, _ = make_predictor()
        with self.assertRaisesRegex(PreprocessingConfigurationError, "전처리기"):
            predictor.predict(
                newtown="gyosan", total_population=100000, age_ratios=VALID_RATIOS
            )

    def test_filtered_json_output_and_model_reuse(self) -> None:
        first = self.predictor.predict(
            newtown="gyosan", total_population=100000, age_ratios=VALID_RATIOS
        )
        second = self.predictor.predict(
            newtown="gyosan", total_population=100000, age_ratios=VALID_RATIOS
        )
        self.assertEqual(self.runner.forward_calls, 2)
        self.assertEqual(first, second)
        self.assertEqual(len(first["od"]), 5)
        self.assertNotIn("self_loop_policy", first["metadata"])
        json.dumps(first, ensure_ascii=False, allow_nan=False)

    def test_shape_dtype_and_mask_contract(self) -> None:
        base = make_inputs()
        with self.assertRaisesRegex(TensorShapeError, "a_spatial"):
            self.predictor.predict_from_tensors(
                replace(base, a_spatial=torch.zeros(3, 4))
            )
        with self.assertRaisesRegex(TensorShapeError, "mask는"):
            self.predictor.predict_from_tensors(
                replace(base, mask=torch.tensor([True, False]))
            )
        with self.assertRaisesRegex(TensorShapeError, "code 수"):
            self.predictor.predict_from_tensors(
                replace(base, city_codes=("NODE_0", "NODE_1"))
            )
        with self.assertRaisesRegex(TensorShapeError, "dtype"):
            self.predictor.predict_from_tensors(
                replace(base, mask=base.mask.to(torch.int64))
            )
        bad_od = base.x_od_masked.clone()
        bad_od[0, 1] = 1
        with self.assertRaisesRegex(PreprocessingConfigurationError, "mask=True.*행과 열"):
            self.predictor.predict_from_tensors(replace(base, x_od_masked=bad_od))

    def test_missing_internal_checkpoint_is_explicit(self) -> None:
        with self.assertRaisesRegex(CheckpointLoadError, "파일이 없습니다"):
            MAEPredictor(weight_file_name="missing.pth")


class OutputAdapterTests(unittest.TestCase):
    def test_duplicate_codes_are_summed_and_external_pair_is_excluded(self) -> None:
        rows = ODOutputAdapter().adapt(
            torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
            origin_codes=("ZONE", "ZONE", "EXTERNAL"),
            destination_codes=("ZONE", "ZONE", "EXTERNAL"),
            newtown_zone_codes=("ZONE",),
        )
        by_pair = {
            (row["origin_code"], row["destination_code"]): row["predicted_trips"]
            for row in rows
        }
        self.assertEqual(by_pair[("ZONE", "ZONE")], 12.0)
        self.assertEqual(by_pair[("ZONE", "EXTERNAL")], 9.0)
        self.assertEqual(by_pair[("EXTERNAL", "ZONE")], 15.0)
        self.assertNotIn(("EXTERNAL", "EXTERNAL"), by_pair)


class RepositoryModelSmokeTests(unittest.TestCase):
    def test_bundled_checkpoint_cpu_forward(self) -> None:
        root = Path(__file__).resolve().parents[1]
        self.assertTrue((root / "model" / "mae.pth").is_file())

        predictor = MAEProvider(device="cpu")
        node_count = 4
        codes = tuple(f"NODE_{index:04d}" for index in range(node_count))
        inputs = ModelInputs(
            x_static=torch.zeros(node_count, 20),
            x_od_masked=torch.zeros(node_count, node_count),
            x_dist=torch.zeros(node_count, node_count),
            a_spatial=torch.zeros(node_count, node_count),
            mask=torch.tensor([True, False, False, False]),
            city_codes=codes,
            newtown_zone_codes=(codes[0],),
            population_allocation_method="repository_model_smoke",
        )
        result = predictor.predict_from_tensors(
            inputs, request_metadata={"newtown": "smoke"}
        )

        self.assertFalse(predictor.model.training)
        self.assertEqual(predictor.num_features, 20)
        self.assertEqual(result["metadata"]["checkpoint"], "mae.pth")
        self.assertEqual(result["metadata"]["node_count"], node_count)
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
