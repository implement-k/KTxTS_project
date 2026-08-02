import math
import importlib.util
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mae_year_loss_search.losses import available_losses, build_loss, loss_definition
from mae_year_loss_search.runner import (
    _cpu_byte_rng_state,
    _save_checkpoint,
    aggregate_validation,
    canonical_modules,
    capture_rng_state,
    data_manifest,
    fingerprint,
    make_train_loader,
    restore_rng_state,
    seed_everything,
)


DATASET_MODULE, MODELS_MODULE, VALIDATION_MODULE = canonical_modules()
_UPSTREAM_SPEC = importlib.util.spec_from_file_location(
    "canonical_mae_year_upstream_loss", ROOT / "src" / "mae-year" / "loss.py"
)
if _UPSTREAM_SPEC is None or _UPSTREAM_SPEC.loader is None:
    raise ImportError("cannot load canonical mae-year loss.py")
UPSTREAM_LOSS = importlib.util.module_from_spec(_UPSTREAM_SPEC)
_UPSTREAM_SPEC.loader.exec_module(UPSTREAM_LOSS)


class LossParityTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.target = torch.rand(2, 4, 4) * 3.0
        self.node_mask = torch.tensor(
            [[True, False, False, False], [False, True, False, False]]
        )
        self.active = torch.tensor(
            [[True, True, True, False], [True, True, False, True]]
        )
        cells = self.node_mask.unsqueeze(1) | self.node_mask.unsqueeze(2)
        self.cells = cells & self.active.unsqueeze(1) & self.active.unsqueeze(2)

    def _compare_upstream(self, name, upstream):
        alpha = 3.25
        left = torch.randn(2, 4, 4, requires_grad=True)
        right = left.detach().clone().requires_grad_(True)
        ours = build_loss(name)(
            left,
            self.target,
            mask=self.node_mask,
            active_node_mask=self.active,
            current_alpha=alpha,
        ).total
        expected = upstream(right, self.target, alpha, self.cells)
        torch.testing.assert_close(ours, expected, rtol=0, atol=0)
        ours.backward()
        expected.backward()
        torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)

    def test_weighted_mse_matches_upstream_loss_and_gradient_exactly(self):
        self._compare_upstream("weighted_mse", UPSTREAM_LOSS.WeightedMSELoss())

    def test_hybrid_matches_upstream_loss_and_gradient_exactly(self):
        self._compare_upstream(
            "hybrid_weighted_mse", UPSTREAM_LOSS.HybridWeightedMSELoss()
        )

    def test_hybrid_components_are_separate_and_sum_to_total(self):
        prediction = torch.randn(2, 4, 4, requires_grad=True)
        output = build_loss("hybrid_weighted_mse")(
            prediction,
            self.target,
            mask=self.node_mask,
            active_node_mask=self.active,
            current_alpha=2.0,
        )
        self.assertEqual(
            set(output.components), {"weighted_mse", "raw_huber", "scaled_raw_huber"}
        )
        torch.testing.assert_close(
            output.total,
            output.components["weighted_mse"] + output.components["scaled_raw_huber"],
        )

    def test_all_registry_losses_have_finite_forward_backward(self):
        self.assertEqual(
            available_losses(), ("weighted_mse", "hybrid_weighted_mse", "huber")
        )
        for name in available_losses():
            prediction = torch.zeros(2, 4, 4, requires_grad=True)
            output = build_loss(name)(
                prediction,
                torch.zeros_like(prediction),
                mask=self.node_mask,
                active_node_mask=self.active,
                current_alpha=1.0,
            )
            self.assertTrue(torch.isfinite(output.total), name)
            output.total.backward()
            self.assertTrue(torch.isfinite(prediction.grad).all(), name)

    def test_empty_mask_returns_differentiable_zero(self):
        for name in available_losses():
            prediction = torch.randn(1, 3, 3, requires_grad=True)
            output = build_loss(name)(
                prediction,
                torch.zeros_like(prediction),
                mask=torch.zeros(1, 3, dtype=torch.bool),
                active_node_mask=torch.ones(1, 3, dtype=torch.bool),
            )
            self.assertEqual(float(output.total.detach()), 0.0)
            output.total.backward()
            torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction))

    def test_inactive_nodes_receive_no_external_gradient(self):
        for name in available_losses():
            prediction = torch.randn(1, 4, 4, requires_grad=True)
            active = torch.tensor([[True, True, False, True]])
            output = build_loss(name)(
                prediction,
                torch.ones_like(prediction),
                mask=torch.ones(1, 4, dtype=torch.bool),
                active_node_mask=active,
                current_alpha=2.0,
            )
            output.total.backward()
            self.assertEqual(float(prediction.grad[:, 2, :].abs().sum()), 0.0)
            self.assertEqual(float(prediction.grad[:, :, 2].abs().sum()), 0.0)

    def test_upstream_huber_accepts_shared_train_loop_signature(self):
        prediction = torch.randn(1, 3, 3, requires_grad=True)
        target = torch.zeros_like(prediction)
        mask = torch.ones_like(prediction, dtype=torch.bool)
        loss = UPSTREAM_LOSS.HuberLoss()(prediction, target, 4.0, mask)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_upstream_huber_retains_legacy_three_argument_mask_call(self):
        prediction = torch.randn(1, 3, 3)
        target = torch.zeros_like(prediction)
        mask = torch.zeros_like(prediction, dtype=torch.bool)
        mask[:, 0, 0] = True
        expected = torch.nn.functional.huber_loss(
            prediction[mask], target[mask], delta=1.0, reduction="mean"
        )
        actual = UPSTREAM_LOSS.HuberLoss()(prediction, target, mask)
        torch.testing.assert_close(actual, expected)


class CanonicalModelTest(unittest.TestCase):
    def test_runner_preserves_upstream_disabled_mha_fastpath(self):
        if hasattr(torch.backends, "mha"):
            self.assertFalse(torch.backends.mha.get_fastpath_enabled())

    def test_small_forward_uses_input_device_for_diagonal_indices(self):
        model = MODELS_MODULE.ODMAE(
            num_features=5, d_model=8, nhead=2, num_layers=1
        )
        batch, nodes = 1, 4
        output = model(
            torch.randn(batch, nodes, 5),
            torch.zeros(batch, nodes, nodes),
            torch.zeros(batch, nodes, nodes),
            torch.zeros(batch, nodes, nodes),
            torch.tensor([[True, False, False, False]]),
            torch.ones(batch, nodes, dtype=torch.bool),
        )
        self.assertEqual(output.shape, (batch, nodes, nodes))
        self.assertTrue(torch.isfinite(output).all())


class ValidationContractTest(unittest.TestCase):
    @staticmethod
    def _metadata(samples=1):
        return {"city": {task: [{} for _ in range(samples)] for task in range(5)}}

    def test_task_zero_through_four_completeness(self):
        VALIDATION_MODULE.validate_metadata_completeness(self._metadata())
        broken = self._metadata()
        broken["city"][3] = []
        with self.assertRaisesRegex(ValueError, r"tasks \[3\]"):
            VALIDATION_MODULE.validate_metadata_completeness(broken)

    def test_official_year_metadata_has_all_cities_tasks_and_samples(self):
        for year in ("2019", "2023"):
            path = ROOT / "dataset" / "fixed_eval" / f"fixed_val_meta_{year}.pt"
            metadata = torch.load(path, map_location="cpu", weights_only=False)
            VALIDATION_MODULE.validate_metadata_completeness(metadata)
            self.assertEqual(set(metadata), {"검단", "동탄", "위례"})
            self.assertTrue(
                all(len(tasks[task]) == 50 for tasks in metadata.values() for task in range(5))
            )

    def test_validation_sample_exception_is_explicit(self):
        sample = {
            "X_static": torch.zeros(2, 3),
            "X_dist": torch.zeros(2, 2),
            "mask": torch.tensor([True, False]),
            "X_OD_masked": torch.zeros(2, 2),
            "A_spatial": torch.zeros(2, 2),
            "active_node_mask": torch.ones(2, dtype=torch.bool),
            "y_OD_raw": torch.zeros(2, 2),
        }

        class Broken(torch.nn.Module):
            def forward(self, *args):
                raise ValueError("deliberate")

        args = (
            Broken(), {"test_indices": []}, "2023", "city", 0, "val", [0], [],
            torch.device("cpu"),
        )
        with mock.patch.object(
            VALIDATION_MODULE, "apply_merge_events", return_value=sample
        ):
            with self.assertRaisesRegex(RuntimeError, "2023/city/task=0"):
                VALIDATION_MODULE._eval_one_sample(args)

    def test_validation_expm1_guard_and_float64_rmse_are_finite(self):
        sample = {
            "X_static": torch.zeros(2, 3),
            "X_dist": torch.zeros(2, 2),
            "mask": torch.tensor([True, False]),
            "X_OD_masked": torch.zeros(2, 2),
            "A_spatial": torch.zeros(2, 2),
            "active_node_mask": torch.ones(2, dtype=torch.bool),
            "y_OD_raw": torch.ones(2, 2),
        }

        class Extreme(torch.nn.Module):
            def forward(self, *args):
                return torch.full((1, 2, 2), 100.0)

        args = (
            Extreme(), {"test_indices": []}, "2023", "city", 0, "val", [0], [],
            torch.device("cpu"),
        )
        with mock.patch.object(
            VALIDATION_MODULE, "apply_merge_events", return_value=sample
        ):
            record = VALIDATION_MODULE._eval_one_sample(args)
        for key in ("cpc", "rmse", "prmse", "offdiag_rmse", "diagonal_rmse"):
            self.assertTrue(math.isfinite(record[key]), key)
        self.assertEqual(record["overall_cell_count"], 3)
        self.assertEqual(record["offdiag_cell_count"], 2)
        self.assertEqual(record["diagonal_cell_count"], 1)

    def test_aggregate_preserves_year_and_all_task_counts(self):
        records = []
        for task in range(5):
            records.append(
                {
                    "year": "2019",
                    "city": "city",
                    "task": task,
                    **{key: 1.0 for key in (
                        "cpc", "rmse", "prmse", "offdiag_cpc", "offdiag_rmse",
                        "offdiag_prmse", "diagonal_cpc", "diagonal_rmse", "diagonal_prmse",
                    )},
                    "overall_cell_count": 3,
                    "offdiag_cell_count": 2,
                    "diagonal_cell_count": 1,
                }
            )
        result = aggregate_validation(records, "2019")
        self.assertEqual(result["mean"]["year"], "2019")
        self.assertEqual(result["mean"]["task_sample_counts"], {str(i): 1 for i in range(5)})
        self.assertEqual(len(result["groups"]), 5)


class RandomDataset(Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, index):
        return torch.tensor(
            [index, np.random.randint(0, 1_000_000), random.randrange(1_000_000)]
        )


class CheckpointAndFingerprintTest(unittest.TestCase):
    @staticmethod
    def _args():
        return SimpleNamespace(seed=123, batch_size=2, num_workers=0, device="cpu")

    def test_uninterrupted_and_restored_rng_dataloader_order_match(self):
        args = self._args()
        seed_everything(args.seed, True, torch.device("cpu"))
        loader = make_train_loader(RandomDataset(), args)
        list(iter(loader))
        state = capture_rng_state(loader)
        expected = [batch.clone() for batch in loader]

        random.seed(999)
        np.random.seed(999)
        torch.manual_seed(999)
        restored = make_train_loader(RandomDataset(), args)
        restore_rng_state(state, restored)
        actual = [batch.clone() for batch in restored]
        self.assertEqual(len(expected), len(actual))
        for left, right in zip(expected, actual):
            torch.testing.assert_close(left, right)

    def test_checkpoint_mapped_rng_state_is_normalized_to_cpu_bytes(self):
        state = torch.arange(16, dtype=torch.uint8)[::2]
        normalized = _cpu_byte_rng_state(state, "test")
        self.assertEqual(normalized.device.type, "cpu")
        self.assertEqual(normalized.dtype, torch.uint8)
        self.assertTrue(normalized.is_contiguous())

    def test_checkpoint_saves_and_restores_training_state(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        output = model(torch.ones(1, 2)).sum()
        output.backward()
        optimizer.step()
        scheduler.step()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            _save_checkpoint(
                path,
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
            )
            loaded = torch.load(path, map_location="cpu", weights_only=False)
        restored = torch.nn.Linear(2, 1)
        restored.load_state_dict(loaded["model"])
        for left, right in zip(model.parameters(), restored.parameters()):
            torch.testing.assert_close(left, right)
        self.assertEqual(loaded["scheduler"]["last_epoch"], 1)

    def test_fingerprint_separates_year_loss_and_parameters(self):
        revision = {"git_sha": "abc", "code_manifest_hash": "def"}
        data = [{"path": "year", "size": 1, "sha256": "hash"}]
        base = {"year": "2023", "loss": loss_definition("weighted_mse")}
        first = fingerprint(base, data, revision)
        self.assertNotEqual(first, fingerprint({**base, "year": "2019"}, data, revision))
        self.assertNotEqual(
            first,
            fingerprint(
                {"year": "2023", "loss": loss_definition("huber")}, data, revision
            ),
        )
        self.assertNotEqual(
            fingerprint(
                {"year": "2023", "loss": loss_definition("huber", {"delta": 1.0})},
                data,
                revision,
            ),
            fingerprint(
                {"year": "2023", "loss": loss_definition("huber", {"delta": 2.0})},
                data,
                revision,
            ),
        )

    def test_data_manifest_is_year_specific_and_excludes_fixed_test(self):
        for year in ("2019", "2023"):
            paths = [row["path"] for row in data_manifest(year)]
            self.assertTrue(all("fixed_test" not in path for path in paths))
            self.assertTrue(any(year in path for path in paths))


if __name__ == "__main__":
    unittest.main()
