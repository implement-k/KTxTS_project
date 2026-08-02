"""Reproducible, year-isolated loss-search runner for canonical mae-year."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .losses import available_losses, build_loss, loss_definition


ROOT = Path(__file__).resolve().parents[2]
MAE_YEAR_DIR = ROOT / "src" / "mae-year"
DATA_DIR = ROOT / "dataset"
GRAD_CLIP_THRESHOLD = 1.0
METRIC_KEYS = (
    "cpc",
    "rmse",
    "prmse",
    "offdiag_cpc",
    "offdiag_rmse",
    "offdiag_prmse",
    "diagonal_cpc",
    "diagonal_rmse",
    "diagonal_prmse",
)
CELL_COUNT_KEYS = (
    "overall_cell_count",
    "offdiag_cell_count",
    "diagonal_cell_count",
)


def _json_object(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("loss parameters must be a JSON object")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", choices=("2019", "2023"), required=True)
    parser.add_argument("--loss", choices=available_losses(), required=True)
    parser.add_argument("--loss-params", type=_json_object, default={})
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=_positive_int, default=70)
    parser.add_argument("--batch-size", type=_positive_int, default=32)
    parser.add_argument("--max-train-batches", type=_positive_int)
    parser.add_argument("--validation-every", type=_positive_int, default=2)
    parser.add_argument("--fixed-max-samples-per-group", type=_positive_int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "experiments" / "mae_year_loss_search",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--num-workers", type=_nonnegative_int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--alpha-start", type=float, default=1.0)
    parser.add_argument("--alpha-end", type=float, default=10.0)
    return parser.parse_args(argv)


def _load_module(name: str, path: Path) -> ModuleType:
    module = sys.modules.get(name)
    if module is not None:
        return module
    src_dir = str(ROOT / "src")
    mae_dir = str(MAE_YEAR_DIR)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    if mae_dir not in sys.path:
        sys.path.insert(0, mae_dir)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def canonical_modules() -> tuple[ModuleType, ModuleType, ModuleType]:
    """Load, without copying, the canonical mae-year dataset/model/validation."""
    # Canonical train.py disables the MHA inference fast path.  Preserve that
    # runtime contract: with its additive distance-bias mask, the fast path can
    # produce all-NaN eval output even when the training path is finite.
    if hasattr(torch.backends, "mha"):
        torch.backends.mha.set_fastpath_enabled(False)
    dataset = _load_module("canonical_mae_year_dataset", MAE_YEAR_DIR / "dataset.py")
    models = _load_module("canonical_mae_year_models", MAE_YEAR_DIR / "models.py")
    validation = _load_module(
        "canonical_mae_year_validation", MAE_YEAR_DIR / "validation.py"
    )
    return dataset, models, validation


def _set_cuda_backend(name: str, enabled: bool) -> bool:
    setter = getattr(torch.backends.cuda, f"enable_{name}_sdp", None)
    if setter is None:
        return False
    setter(enabled)
    return True


def _cuda_backend_enabled(name: str):
    getter = getattr(torch.backends.cuda, f"{name}_sdp_enabled", None)
    return getter() if getter is not None else None


def configure_determinism(
    deterministic: bool, device: torch.device | None = None
) -> dict[str, Any]:
    if deterministic and device is not None and device.type == "cuda":
        cublas_config = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if cublas_config not in {":4096:8", ":16:8"}:
            raise RuntimeError(
                "strict deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG "
                f"to be :4096:8 or :16:8, got {cublas_config!r}"
            )
    torch.use_deterministic_algorithms(deterministic, warn_only=False)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic

    requested = {
        "flash": not deterministic,
        "mem_efficient": not deterministic,
        "math": True,
        "cudnn": not deterministic,
    }
    supported = {
        name: _set_cuda_backend(name, enabled) for name, enabled in requested.items()
    }
    active = {name: _cuda_backend_enabled(name) for name in requested}
    if deterministic and device is not None and device.type == "cuda":
        required = ("flash", "mem_efficient", "math")
        if not all(supported[name] for name in required):
            missing = [name for name in required if not supported[name]]
            raise RuntimeError(
                "strict deterministic CUDA requires explicit SDPA controls; "
                f"missing: {missing}"
            )
        if active["flash"] or active["mem_efficient"] or not active["math"]:
            raise RuntimeError("failed to enforce math-only deterministic CUDA SDPA")
    warn_only_getter = getattr(
        torch, "is_deterministic_algorithms_warn_only_enabled", None
    )
    return {
        "requested": deterministic,
        "algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "warn_only": warn_only_getter() if warn_only_getter is not None else False,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn_deterministic": getattr(torch.backends.cudnn, "deterministic", None),
        "cudnn_benchmark": getattr(torch.backends.cudnn, "benchmark", None),
        "sdpa": {
            "policy": "math_only" if deterministic else "performance_backends_allowed",
            "requested": requested,
            "control_supported": supported,
            "active": active,
        },
    }


def seed_everything(seed: int, deterministic: bool, device: torch.device) -> dict:
    state = configure_determinism(deterministic, device)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return state


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def make_train_loader(dataset, args) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=resolve_device(args.device).type == "cuda",
        generator=generator,
        worker_init_fn=seed_worker,
    )
    loader.mae_year_generator = generator
    return loader


def capture_rng_state(train_loader: DataLoader) -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "dataloader_generator": train_loader.mae_year_generator.get_state(),
    }


def _cpu_byte_rng_state(value: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} RNG state must be a tensor")
    value = value.detach().cpu()
    if value.dtype != torch.uint8:
        raise TypeError(f"{name} RNG state must be uint8")
    return value.contiguous()


def restore_rng_state(state: dict, train_loader: DataLoader) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(_cpu_byte_rng_state(state["torch"], "PyTorch CPU"))
    saved_cuda = state.get("cuda")
    if torch.cuda.is_available() and saved_cuda is not None:
        if len(saved_cuda) != torch.cuda.device_count():
            raise RuntimeError("checkpoint CUDA RNG device count does not match runtime")
        torch.cuda.set_rng_state_all(
            [
                _cpu_byte_rng_state(value, f"CUDA device {index}")
                for index, value in enumerate(saved_cuda)
            ]
        )
    train_loader.mae_year_generator.set_state(
        _cpu_byte_rng_state(state["dataloader_generator"], "DataLoader generator")
    )


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable; no fallback")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("--device mps requested but MPS is unavailable; no fallback")
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def clip_gradients(parameters, threshold: float = GRAD_CLIP_THRESHOLD) -> float:
    norm = torch.nn.utils.clip_grad_norm_(
        parameters, max_norm=threshold, error_if_nonfinite=True
    )
    value = float(norm.detach().cpu())
    if not np.isfinite(value):
        raise FloatingPointError(f"non-finite pre-clip gradient norm: {value}")
    return value


def gradient_diagnostics(norms: list[float], threshold: float) -> dict:
    clipped = sum(norm > threshold for norm in norms)
    return {
        "grad_norm_preclip_mean": float(np.mean(norms)) if norms else None,
        "grad_norm_preclip_max": float(np.max(norms)) if norms else None,
        "grad_clip_threshold": threshold,
        "grad_clipped_steps": clipped,
        "grad_total_steps": len(norms),
        "grad_clipped_fraction": clipped / len(norms) if norms else 0.0,
    }


def start_epoch_measurement(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    return time.perf_counter()


def finish_epoch_measurement(started: float, device: torch.device) -> dict:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
    else:
        peak_allocated = None
        peak_reserved = None
    return {
        "epoch_elapsed_seconds": time.perf_counter() - started,
        "cuda_peak_allocated_bytes": peak_allocated,
        "cuda_peak_reserved_bytes": peak_reserved,
    }


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(ROOT), *args], check=False, capture_output=True, text=True
    )
    return result.stdout.strip()


def _file_info(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path.relative_to(ROOT)),
        "size": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def data_manifest(year: str) -> list[dict]:
    dong_name = "OD_dong_list_2019_unique.xlsx" if year == "2019" else "OD_dong_list_2023.xlsx"
    od_csv = DATA_DIR / f"od_data_{year}.csv"
    od_source = od_csv if od_csv.exists() else DATA_DIR / "fixed_eval" / f"base_data_{year}.pt"
    paths = [
        DATA_DIR / f"dist_data_{year}.csv",
        DATA_DIR / f"final_static_features_{year}.csv",
        DATA_DIR / "raw" / "dong" / dong_name,
        MAE_YEAR_DIR / f"merge_cache_{year}.pkl",
        DATA_DIR / "fixed_eval" / f"fixed_val_meta_{year}.pt",
        od_source,
    ]
    return [_file_info(path) for path in paths]


def code_manifest() -> list[dict]:
    paths = [
        ROOT / "src" / "config.py",
        ROOT / "src" / "evaluation" / "fixed_eval_utils.py",
        MAE_YEAR_DIR / "dataset.py",
        MAE_YEAR_DIR / "models.py",
        MAE_YEAR_DIR / "loss.py",
        MAE_YEAR_DIR / "train.py",
        MAE_YEAR_DIR / "validation.py",
        Path(__file__),
        Path(__file__).with_name("losses.py"),
        ROOT / "scripts" / "run_mae_year_loss_search.py",
    ]
    return [_file_info(path) for path in paths]


def manifest_hash(manifest: list[dict]) -> str:
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def fingerprint(config: dict, data: list[dict], revision: dict) -> str:
    payload = json.dumps(
        {"config": config, "data": data, "revision": revision},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _save_checkpoint(path: Path, state: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def aggregate_validation(records: list[dict], year: str) -> dict:
    if not records:
        raise ValueError("fixed validation produced no records")
    expected_groups = {(city, task) for city in {row["city"] for row in records} for task in range(5)}
    actual_groups = {(row["city"], row["task"]) for row in records}
    if actual_groups != expected_groups:
        raise RuntimeError(
            f"Task 0-4 completeness failure: expected {sorted(expected_groups)}, "
            f"got {sorted(actual_groups)}"
        )

    groups = []
    for city, task in sorted(actual_groups, key=lambda item: (item[0], item[1])):
        selected = [row for row in records if row["city"] == city and row["task"] == task]
        group = {
            "year": year,
            "city": city,
            "task": task,
            "sample_count": len(selected),
        }
        for key in METRIC_KEYS:
            group[key] = float(np.mean([row[key] for row in selected]))
        for key in CELL_COUNT_KEYS:
            group[key] = int(sum(row[key] for row in selected))
        groups.append(group)

    mean = {key: float(np.mean([row[key] for row in records])) for key in METRIC_KEYS}
    mean.update(
        {
            "year": year,
            "sample_count": len(records),
            "group_count": len(groups),
            **{key: int(sum(row[key] for row in records)) for key in CELL_COUNT_KEYS},
            "task_sample_counts": {
                str(task): sum(row["task"] == task for row in records)
                for task in range(5)
            },
        }
    )
    return {"year": year, "mean": mean, "groups": groups, "samples": records}


def save_validation_result(result: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "fixed_validation.json", result)
    _write_csv(output_dir / "fixed_validation_groups.csv", result["groups"])
    _write_csv(output_dir / "fixed_validation_samples.csv", result["samples"])


def environment_metadata(device: torch.device, determinism: dict) -> dict:
    def version(package: str):
        try:
            return importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            return None

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scikit_learn": version("scikit-learn"),
        "device": str(device),
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "determinism": determinism,
    }


def experiment_config(args, resolved_loss: dict, model_config: dict) -> dict:
    return {
        "year": args.year,
        "seed": args.seed,
        "model": model_config,
        "dataset_preprocessing": {
            "implementation": "src/mae-year/dataset.py:ODDataset",
            "static_normalize": True,
            "od_log1p": True,
            "distance_log1p": True,
            "train_excludes_same_year_validation_and_test_city_indices": True,
        },
        "masking_merge": {
            "stratified_masking": True,
            "merge_augmentation": True,
            "min_mask_size": 3,
            "max_mask_size": 150,
            "mask_curriculum": "linear_by_epoch",
            "active_node_contract": "both OD endpoints must be active",
        },
        "loss": resolved_loss,
        "alpha_curriculum": {
            "start": args.alpha_start,
            "end": args.alpha_end,
            "schedule": "linear_by_epoch",
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "max_train_batches": args.max_train_batches,
            "learning_rate": args.learning_rate,
            "max_learning_rate": args.max_learning_rate,
            "weight_decay": args.weight_decay,
            "gradient_clip_norm": GRAD_CLIP_THRESHOLD,
            "deterministic": args.deterministic,
            "num_workers": args.num_workers,
            "device": args.device,
        },
        "validation": {
            "split": "fixed_val_only",
            "year": args.year,
            "tasks": [0, 1, 2, 3, 4],
            "every_epochs": args.validation_every,
            "max_samples_per_group": args.fixed_max_samples_per_group,
            "checkpoint_selection": {
                "best_cpc": "same-year mean sample CPC, maximize",
                "best_rmse": "same-year mean sample RMSE, minimize",
            },
        },
    }


def _alpha(args, epoch: int) -> float:
    progress = epoch / max(1, args.epochs - 1)
    return args.alpha_start + (args.alpha_end - args.alpha_start) * progress


def run(args) -> Path:
    started = time.time()
    device = resolve_device(args.device)
    determinism = seed_everything(args.seed, args.deterministic, device)
    dataset_module, models_module, validation_module = canonical_modules()
    dataset = dataset_module.ODDataset(year=args.year)

    model_config = {
        "num_features": int(dataset.X_static.shape[1]),
        "d_model": 128,
        "nhead": 8,
        "num_layers": 4,
        "od_embed_layers": 3,
        "use_distance_friction": True,
        "use_self_loop_predictor": True,
        "use_mask_channel": False,
    }
    resolved_loss = loss_definition(args.loss, args.loss_params)
    config = experiment_config(args, resolved_loss, model_config)
    code = code_manifest()
    revision = {
        "git_sha": _git("rev-parse", "HEAD"),
        "code_manifest_hash": manifest_hash(code),
        "code_files": [{"path": row["path"], "sha256": row["sha256"]} for row in code],
    }
    data = data_manifest(args.year)
    run_fingerprint = fingerprint(config, data, revision)
    run_dir = args.output_root / args.year / args.loss / run_fingerprint
    checkpoint_dir = run_dir / "checkpoints"
    completed_path = run_dir / "completed.json"
    if completed_path.exists():
        completed = json.loads(completed_path.read_text(encoding="utf-8"))
        if completed.get("fingerprint") != run_fingerprint:
            raise ValueError(f"completed metadata fingerprint mismatch: {completed_path}")
        print(f"[skip] completed experiment: {run_dir}")
        return run_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "fingerprint": run_fingerprint,
        "config": config,
        "git": {"sha": revision["git_sha"], "status": _git("status", "--short")},
        "code_manifest_hash": revision["code_manifest_hash"],
        "code_manifest": code,
        "data_manifest": data,
        "data_hash": manifest_hash(data),
        "environment": environment_metadata(device, determinism),
        "started_unix": started,
    }
    _write_json(run_dir / "metadata.json", metadata)

    train_loader = make_train_loader(dataset, args)
    model = models_module.ODMAE(**model_config).to(device)
    criterion = build_loss(args.loss, **resolved_loss["parameters"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    batches_per_epoch = min(
        len(train_loader), args.max_train_batches or len(train_loader)
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.max_learning_rate,
        total_steps=max(1, args.epochs * batches_per_epoch),
    )

    from evaluation.fixed_eval_utils import make_base_data

    fixed_base = make_base_data(dataset)
    fixed_meta_path = DATA_DIR / "fixed_eval" / f"fixed_val_meta_{args.year}.pt"
    fixed_meta = torch.load(fixed_meta_path, map_location="cpu", weights_only=False)
    validation_module.validate_metadata_completeness(fixed_meta)

    latest_path = checkpoint_dir / "latest.pt"
    start_epoch = 0
    best_cpc = float("-inf")
    best_rmse = float("inf")
    epoch_rows: list[dict] = []
    diagnostics_history: list[dict] = []
    last_validation = None
    elapsed_before_resume = 0.0
    if latest_path.exists():
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        if checkpoint["fingerprint"] != run_fingerprint:
            raise ValueError("latest checkpoint fingerprint does not match this run")
        if checkpoint["loss_definition"] != resolved_loss:
            raise ValueError("latest checkpoint loss definition does not match this run")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        restore_rng_state(checkpoint["rng_state"], train_loader)
        start_epoch = int(checkpoint["epoch"]) + 1
        best_cpc = float(checkpoint["best_fixed_val_cpc"])
        best_rmse = float(checkpoint["best_fixed_val_rmse"])
        epoch_rows = checkpoint.get("epoch_rows", [])
        diagnostics_history = checkpoint.get("diagnostics_history", [])
        last_validation = checkpoint.get("last_validation")
        elapsed_before_resume = float(checkpoint.get("elapsed_seconds", 0.0))
        print(f"[resume] {latest_path} at epoch {start_epoch + 1}")
    elif args.resume:
        raise FileNotFoundError(f"--resume requested but no checkpoint exists: {latest_path}")

    for epoch in range(start_epoch, args.epochs):
        epoch_started = start_epoch_measurement(device)
        progress = epoch / max(1, args.epochs - 1)
        dataset.max_mask_size = int(3 + (150 - 3) * progress)
        current_alpha = _alpha(args, epoch)
        model.train()
        component_sums: dict[str, float] = {}
        gradient_norms: list[float] = []
        executed_batches = 0
        for batch_index, batch in enumerate(train_loader):
            if batch_index >= batches_per_epoch:
                break
            tensors = {name: value.to(device) for name, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            prediction = model(
                tensors["X_static"],
                tensors["X_OD_masked"],
                tensors["X_dist"],
                tensors["A_spatial"],
                tensors["mask"],
                tensors["active_node_mask"],
            )
            output = criterion(
                prediction,
                tensors["y_OD"],
                mask=tensors["loss_mask"],
                active_node_mask=tensors["active_node_mask"],
                current_alpha=current_alpha,
            )
            if not torch.isfinite(output.total):
                raise FloatingPointError(
                    f"non-finite loss at epoch={epoch + 1}, batch={batch_index + 1}"
                )
            output.total.backward()
            gradient_norms.append(clip_gradients(model.parameters()))
            optimizer.step()
            scheduler.step()
            for name, value in output.detached().items():
                component_sums[name] = component_sums.get(name, 0.0) + value
            executed_batches += 1
        if executed_batches == 0:
            raise RuntimeError("no training batches were executed")

        should_validate = (
            (epoch + 1) % args.validation_every == 0 or epoch + 1 == args.epochs
        )
        validation_result = None
        if should_validate:
            records = validation_module.evaluate_and_report(
                base_data=fixed_base,
                val_meta=fixed_meta,
                model=model,
                year_label=args.year,
                split_name="val",
                n_workers=1,
                device=device,
                max_samples_per_group=args.fixed_max_samples_per_group,
            )
            validation_result = aggregate_validation(records, args.year)
            last_validation = validation_result
            save_validation_result(
                validation_result,
                run_dir / "validation" / f"epoch_{epoch + 1:04d}",
            )

        diagnostics = {
            "epoch": epoch + 1,
            **gradient_diagnostics(gradient_norms, GRAD_CLIP_THRESHOLD),
            **finish_epoch_measurement(epoch_started, device),
        }
        diagnostics_history.append(diagnostics)
        row = {
            "epoch": epoch + 1,
            "alpha": current_alpha,
            "mask_size_max": dataset.max_mask_size,
            **{
                f"loss_{name}": value / executed_batches
                for name, value in component_sums.items()
            },
            **{key: value for key, value in diagnostics.items() if key != "epoch"},
        }
        if validation_result is not None:
            row.update(
                {
                    f"fixed_val_{key}": value
                    for key, value in validation_result["mean"].items()
                    if not isinstance(value, dict)
                }
            )
        epoch_rows.append(row)

        is_best_cpc = False
        is_best_rmse = False
        if validation_result is not None:
            current_cpc = validation_result["mean"]["cpc"]
            current_rmse = validation_result["mean"]["rmse"]
            if np.isfinite(current_cpc) and current_cpc > best_cpc:
                best_cpc = current_cpc
                is_best_cpc = True
            if np.isfinite(current_rmse) and current_rmse < best_rmse:
                best_rmse = current_rmse
                is_best_rmse = True

        checkpoint_state = {
            "fingerprint": run_fingerprint,
            "epoch": epoch,
            "year": args.year,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "rng_state": capture_rng_state(train_loader),
            "model_config": model_config,
            "loss_definition": resolved_loss,
            "alpha_curriculum": config["alpha_curriculum"],
            "config": config,
            "best_fixed_val_cpc": best_cpc,
            "best_fixed_val_rmse": best_rmse,
            "last_validation": last_validation,
            "epoch_rows": epoch_rows,
            "diagnostics_history": diagnostics_history,
            "elapsed_seconds": elapsed_before_resume + time.time() - started,
        }
        _save_checkpoint(latest_path, checkpoint_state)
        if is_best_cpc:
            _save_checkpoint(checkpoint_dir / "best_cpc.pt", checkpoint_state)
        if is_best_rmse:
            _save_checkpoint(checkpoint_dir / "best_rmse.pt", checkpoint_state)
        _write_csv(run_dir / "epochs.csv", epoch_rows)
        print(json.dumps(row, ensure_ascii=False))

    if last_validation is None:
        raise RuntimeError("completed training without a fixed-validation result")
    summary = {
        "fingerprint": run_fingerprint,
        "year": args.year,
        "loss": resolved_loss,
        "best_fixed_val_cpc": best_cpc,
        "best_fixed_val_rmse": best_rmse,
        "final_fixed_validation": last_validation["mean"],
        "elapsed_seconds": elapsed_before_resume + time.time() - started,
    }
    _write_json(run_dir / "summary.json", summary)
    _write_csv(run_dir / "summary.csv", [{
        key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
        for key, value in summary.items()
    }])
    completed = {
        **summary,
        "status": "completed",
        "finished_unix": time.time(),
        "environment": metadata["environment"],
        "git": metadata["git"],
        "code_manifest_hash": metadata["code_manifest_hash"],
        "data_hash": metadata["data_hash"],
        "seed": args.seed,
        "deterministic": determinism,
        "model_config": model_config,
        "alpha_curriculum": config["alpha_curriculum"],
        "diagnostics_history": diagnostics_history,
    }
    _write_json(completed_path, completed)
    return run_dir


def main(argv=None):
    args = parse_args(argv)
    print(run(args))


if __name__ == "__main__":
    main()
