#!/usr/bin/env python3
"""Bounded CPU data smoke for both mae-year datasets and fixed validation."""

import argparse
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.fixed_eval_utils import make_base_data
from mae_year_loss_search.losses import build_loss
from mae_year_loss_search.runner import DATA_DIR, canonical_modules, seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d-model", type=int, default=16)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=1)
    args = parser.parse_args()

    seed_everything(17, True, torch.device("cpu"))
    dataset_module, models_module, validation_module = canonical_modules()
    loaded_paths = []
    original_load = torch.load

    def guarded_load(path, *load_args, **load_kwargs):
        path_text = str(path)
        if "fixed_test" in path_text:
            raise AssertionError(f"fixed-test access is forbidden: {path_text}")
        loaded_paths.append(path_text)
        return original_load(path, *load_args, **load_kwargs)

    torch.load = guarded_load
    try:
        for year in ("2019", "2023"):
            dataset = dataset_module.ODDataset(year=year)
            dataset.max_mask_size = 3
            sample = dataset[0]
            batch = {key: value.unsqueeze(0) for key, value in sample.items()}
            model = models_module.ODMAE(
                num_features=dataset.X_static.shape[1],
                d_model=args.d_model,
                nhead=args.nhead,
                num_layers=args.num_layers,
                od_embed_layers=2,
                use_distance_friction=True,
                use_self_loop_predictor=True,
                use_mask_channel=False,
            )
            prediction = model(
                batch["X_static"], batch["X_OD_masked"], batch["X_dist"],
                batch["A_spatial"], batch["mask"], batch["active_node_mask"],
            )
            output = build_loss("weighted_mse")(
                prediction, batch["y_OD"], mask=batch["loss_mask"],
                active_node_mask=batch["active_node_mask"], current_alpha=1.0,
            )
            output.total.backward()
            if not torch.isfinite(output.total):
                raise FloatingPointError(f"non-finite {year} smoke loss")

            metadata = torch.load(
                DATA_DIR / "fixed_eval" / f"fixed_val_meta_{year}.pt",
                map_location="cpu",
                weights_only=False,
            )
            records = validation_module.evaluate_and_report(
                make_base_data(dataset), metadata, model, year, "val", n_workers=1,
                device=torch.device("cpu"), max_samples_per_group=1,
            )
            task_counts = {task: sum(row["task"] == task for row in records) for task in range(5)}
            task_cells = {
                task: sum(
                    row["overall_cell_count"] for row in records if row["task"] == task
                )
                for task in range(5)
            }
            if not all(task_counts.values()):
                raise AssertionError(f"incomplete {year} validation smoke: {task_counts}")
            print(
                {"year": year, "loss": float(output.total.detach()), "validation": task_counts,
                 "cells": task_cells, "samples": len(records)}
            )
    finally:
        torch.load = original_load

    if any("fixed_test" in path for path in loaded_paths):
        raise AssertionError(f"fixed-test was accessed: {loaded_paths}")
    print({"fixed_test_accessed": False, "torch_load_paths": loaded_paths})


if __name__ == "__main__":
    main()
