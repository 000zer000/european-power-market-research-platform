"""Shared deterministic test configuration."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from power_market.config import Settings, load_settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Return the sample configuration with isolated artifact paths."""
    base = load_settings(Path(__file__).parents[1] / "configs" / "offline.yaml")
    return base.model_copy(update={"project_root": tmp_path})


@pytest.fixture
def compact_settings(settings: Settings) -> Settings:
    """Return a short three-split period suitable for end-to-end tests."""
    data = settings.data.model_copy(
        update={"start_date": date(2023, 1, 1), "end_date": date(2023, 3, 7)}
    )
    splits = settings.splits.model_copy(
        update={
            "train_end": date(2023, 1, 31),
            "validation_end": date(2023, 2, 28),
            "test_end": date(2023, 3, 7),
        }
    )
    forecast = settings.forecast.model_copy(
        update={
            "ridge_alphas": [1.0],
            "tree_learning_rates": [0.1],
            "tree_max_leaf_nodes": [15],
            "tree_max_iter": 20,
            "bootstrap_samples": 20,
        }
    )
    return settings.model_copy(update={"data": data, "splits": splits, "forecast": forecast})
