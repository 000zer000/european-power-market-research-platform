from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from power_market.metrics import (
    daily_block_bootstrap_mae_difference,
    directional_accuracy,
    mae,
    rmse,
)


def test_metric_values() -> None:
    actual = [0.0, 2.0, 4.0]
    predicted = [0.0, 1.0, 7.0]
    assert mae(actual, predicted) == pytest.approx(4 / 3)
    assert rmse(actual, predicted) == pytest.approx(np.sqrt(10 / 3))
    assert directional_accuracy(actual, predicted, [-1, 3, 3]) == pytest.approx(1.0)


def test_metrics_reject_shape_and_nonfinite() -> None:
    with pytest.raises(ValueError, match="identical"):
        mae([1], [1, 2])
    with pytest.raises(ValueError, match="finite"):
        rmse([np.nan], [1])


def test_daily_block_bootstrap_is_deterministic() -> None:
    frame = pd.DataFrame(
        {
            "local_date": ["2025-01-01"] * 2 + ["2025-01-02"] * 2,
            "actual_price": [0, 10, 5, 8],
            "model": [0, 9, 5, 7],
            "reference": [5, 5, 0, 0],
        }
    )
    first = daily_block_bootstrap_mae_difference(frame, "model", "reference", 100, 7)
    second = daily_block_bootstrap_mae_difference(frame, "model", "reference", 100, 7)
    assert first == second
    assert first["ci_95_upper"] < 0
