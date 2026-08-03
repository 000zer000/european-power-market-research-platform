from __future__ import annotations

import json

import pandas as pd

from power_market.pipeline import (
    evaluate_stage,
    feature_stage,
    optimise_stage,
    report_stage,
    run_all,
    train_stage,
    validate_stage,
)


def test_end_to_end_offline_fixture(compact_settings) -> None:
    result = run_all(compact_settings)
    assert result["validation"] == "pass"
    assert result["test_predictions"] == 7 * 24
    assert len(result["figures"]) == 8
    root = compact_settings.project_root
    metrics = json.loads((root / "artifacts" / "metrics" / "forecast_metrics.json").read_text())
    assert set(metrics["overall"]) == {
        "previous_day",
        "previous_week",
        "ridge",
        "hist_gradient_boosting",
    }
    assert (root / "artifacts" / "backtests" / "battery_summary.csv").exists()
    assert (root / "reports" / "research_summary.md").exists()
    run_all_predictions = pd.read_csv(root / "artifacts" / "metrics" / "test_predictions.csv")

    # Exercise the independently callable CLI stages against the same offline evidence.
    assert validate_stage(compact_settings)["status"] == "pass"
    assert len(feature_stage(compact_settings)) == result["observations"]
    assert len(train_stage(compact_settings)) == result["test_predictions"]
    staged_predictions = pd.read_csv(root / "artifacts" / "metrics" / "test_predictions.csv")
    pd.testing.assert_frame_equal(run_all_predictions, staged_predictions)
    assert "overall" in evaluate_stage(compact_settings)
    _, aggregate, _ = optimise_stage(compact_settings)
    assert set(aggregate["strategy"]) >= {"ridge", "perfect_foresight", "do_nothing"}
    assert len(report_stage(compact_settings)) == 9
