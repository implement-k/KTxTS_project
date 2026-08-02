"""Reproducible loss-search infrastructure for canonical ``src/mae-year``."""

from .losses import LossOutput, available_losses, build_loss

__all__ = ["LossOutput", "available_losses", "build_loss"]
