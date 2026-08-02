"""Baseline loss registry for the canonical mae-year log1p OD output."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class LossOutput:
    total: torch.Tensor
    components: dict[str, torch.Tensor]

    def detached(self) -> dict[str, float]:
        values = {"total": float(self.total.detach().cpu())}
        values.update(
            {name: float(value.detach().cpu()) for name, value in self.components.items()}
        )
        return values


def valid_cells(
    prediction: torch.Tensor,
    mask: torch.Tensor | None,
    active_node_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Return the shared masked-cell and active-node intersection."""
    if prediction.ndim != 3 or prediction.shape[-1] != prediction.shape[-2]:
        raise ValueError("prediction must have shape (batch, nodes, nodes)")
    valid = torch.ones_like(prediction, dtype=torch.bool)
    if mask is not None:
        mask = mask.bool()
        if mask.ndim == prediction.ndim - 1:
            if mask.shape != prediction.shape[:2]:
                raise ValueError("node mask must have shape (batch, nodes)")
            mask = mask.unsqueeze(1) | mask.unsqueeze(2)
        if mask.shape != prediction.shape:
            raise ValueError("cell mask must match prediction shape")
        valid &= mask
    if active_node_mask is not None:
        active = active_node_mask.bool()
        if active.shape != prediction.shape[:2]:
            raise ValueError("active_node_mask must have shape (batch, nodes)")
        valid &= active.unsqueeze(1) & active.unsqueeze(2)
    return valid


def _zero(prediction: torch.Tensor) -> torch.Tensor:
    return prediction.sum() * 0.0


class CommonODLoss(nn.Module):
    """All candidates consume log1p prediction/target and one mask contract."""

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
        active_node_mask: torch.Tensor | None = None,
        current_alpha: float = 1.0,
    ) -> LossOutput:
        raise NotImplementedError


class WeightedMSELoss(CommonODLoss):
    """Exact mae-year ``WeightedMSELoss`` formula on shared valid cells."""

    def forward(
        self, prediction, target, mask=None, active_node_mask=None, current_alpha=1.0
    ):
        valid = valid_cells(prediction, mask, active_node_mask)
        if not valid.any():
            total = _zero(prediction)
        else:
            pred_valid = prediction[valid]
            target_valid = target[valid]
            weight = 1.0 + current_alpha * target_valid
            total = (((pred_valid - target_valid) ** 2) * weight).mean()
        return LossOutput(total=total, components={"weighted_mse": total})


class HybridWeightedMSELoss(CommonODLoss):
    """Exact mae-year hybrid formula with separately reported terms."""

    def __init__(self, real_penalty_weight: float = 0.005):
        super().__init__()
        if real_penalty_weight < 0:
            raise ValueError("real_penalty_weight must be nonnegative")
        self.real_penalty_weight = float(real_penalty_weight)

    def forward(
        self, prediction, target, mask=None, active_node_mask=None, current_alpha=1.0
    ):
        valid = valid_cells(prediction, mask, active_node_mask)
        if not valid.any():
            weighted_mse = _zero(prediction)
            raw_huber = _zero(prediction)
            total = _zero(prediction)
        else:
            pred_valid = prediction[valid]
            target_valid = target[valid]
            weight = 1.0 + current_alpha * target_valid
            weighted_elements = ((pred_valid - target_valid) ** 2) * weight
            weighted_mse = weighted_elements.mean()
            pred_real = torch.expm1(torch.clamp(pred_valid, max=12.0))
            target_real = torch.expm1(target_valid)
            raw_huber_elements = F.huber_loss(
                pred_real, target_real, delta=100.0, reduction="none"
            )
            raw_huber = raw_huber_elements.mean()
            # Preserve upstream reduction order for bit-exact loss/gradient parity.
            total = (
                weighted_elements + self.real_penalty_weight * raw_huber_elements
            ).mean()
        scaled_raw_huber = self.real_penalty_weight * raw_huber
        return LossOutput(
            total=total,
            components={
                "weighted_mse": weighted_mse,
                "raw_huber": raw_huber,
                "scaled_raw_huber": scaled_raw_huber,
            },
        )


class HuberLoss(CommonODLoss):
    """Plain Huber reference on the mae-year log1p output scale."""

    def __init__(self, delta: float = 1.0):
        super().__init__()
        if delta <= 0:
            raise ValueError("delta must be positive")
        self.delta = float(delta)

    def forward(
        self, prediction, target, mask=None, active_node_mask=None, current_alpha=1.0
    ):
        del current_alpha
        valid = valid_cells(prediction, mask, active_node_mask)
        total = (
            F.huber_loss(
                prediction[valid], target[valid], delta=self.delta, reduction="mean"
            )
            if valid.any()
            else _zero(prediction)
        )
        return LossOutput(total=total, components={"huber": total})


_REGISTRY: dict[str, Callable[..., CommonODLoss]] = {
    "weighted_mse": WeightedMSELoss,
    "hybrid_weighted_mse": HybridWeightedMSELoss,
    "huber": HuberLoss,
}


def available_losses() -> tuple[str, ...]:
    return tuple(_REGISTRY)


def build_loss(name: str, **parameters: Any) -> CommonODLoss:
    try:
        factory = _REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown loss {name!r}; choose from {', '.join(available_losses())}"
        ) from exc
    return factory(**parameters)


def loss_definition(name: str, parameters: dict[str, Any] | None = None) -> dict:
    """Resolve explicit defaults for metadata, checkpoints, and fingerprints."""
    resolved = dict(parameters or {})
    if name == "hybrid_weighted_mse":
        resolved.setdefault("real_penalty_weight", 0.005)
    elif name == "huber":
        resolved.setdefault("delta", 1.0)
    build_loss(name, **resolved)
    formulas = {
        "weighted_mse": "mean((p-t)^2 * (1 + alpha*t))",
        "hybrid_weighted_mse": (
            "mean((p-t)^2*(1+alpha*t)) + real_penalty_weight*"
            "mean(Huber_delta=100(expm1(clamp_max(p,12)), expm1(t)))"
        ),
        "huber": "mean(Huber_delta(p,t))",
    }
    return {
        "name": name,
        "parameters": resolved,
        "prediction_scale": "log1p_od",
        "target_scale": "log1p_od",
        "mask_contract": "masked row-or-column cells intersected with active-node pairs",
        "formula": formulas[name],
    }
