#!/usr/bin/env python3
"""CLI entry point for the canonical mae-year loss-search runner."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mae_year_loss_search.runner import main


if __name__ == "__main__":
    main()
