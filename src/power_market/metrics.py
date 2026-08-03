"""Forecast metrics, slices and daily block-bootstrap uncertainty."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd


def _arrays(actual: Iterable[float], predicted: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    observed = np.asarray(actual, dtype=float)
    forecast = np.asarray(predicted, dtype=float)
    if observed.shape != forecast.shape or observed.size == 0:
        raise ValueError("actual and predicted must be nonempty arrays with identical shapes")
    if not np.isfinite(observed).all() or not np.isfinite(forecast).all():
        raise ValueError("metrics require finite values")
    return observed, forecast


def mae(actual: Iterable[float], predicted: Iterable[float]) -> float:
    """Mean absolute error in the target unit."""
    observed, forecast = _arrays(actual, predicted)
    return float(np.mean(np.abs(observed - forecast)))


def rmse(actual: Iterable[float], predicted: Iterable[float]) -> float:
    """Root mean squared error in the target unit."""
    observed, forecast = _arrays(actual, predicted)
    return float(np.sqrt(np.mean(np.square(observed - forecast))))


def directional_accuracy(
    actual: Iterable[float], predicted: Iterable[float], reference: Iterable[float]
) -> float:
    """Share of correct directions relative to the previous-day reference price."""
    observed, forecast = _arrays(actual, predicted)
    baseline = np.asarray(reference, dtype=float)
    if baseline.shape != observed.shape or not np.isfinite(baseline).all():
        raise ValueError("reference must be finite and match actual")
    relevant = np.sign(observed - baseline) != 0
    if not relevant.any():
        return float("nan")
    return float(
        np.mean(
            np.sign(forecast[relevant] - baseline[relevant])
            == np.sign(observed[relevant] - baseline[relevant])
        )
    )


def daily_block_bootstrap_mae_difference(
    predictions: pd.DataFrame,
    model: str,
    reference: str,
    samples: int,
    seed: int,
) -> dict[str, float | str | int]:
    """Bootstrap whole delivery days for paired MAE(model)-MAE(reference)."""
    errors = predictions.assign(
        model_error=(predictions["actual_price"] - predictions[model]).abs(),
        reference_error=(predictions["actual_price"] - predictions[reference]).abs(),
    )
    daily = errors.groupby("local_date")[["model_error", "reference_error"]].mean()
    differences = (daily["model_error"] - daily["reference_error"]).to_numpy()
    rng = np.random.default_rng(seed)
    draws = rng.choice(differences, size=(samples, len(differences)), replace=True).mean(axis=1)
    return {
        "mean_mae_difference_eur_per_mwh": float(differences.mean()),
        "ci_95_lower": float(np.quantile(draws, 0.025)),
        "ci_95_upper": float(np.quantile(draws, 0.975)),
        "resampling_unit": "local delivery day",
        "bootstrap_samples": samples,
    }


def evaluate_forecasts(
    predictions: pd.DataFrame, samples: int, seed: int, project_root: Path
) -> dict[str, object]:
    """Calculate overall, hourly, monthly, negative and extreme-period errors."""
    models = ["previous_day", "previous_week", "ridge", "hist_gradient_boosting"]
    actual = predictions["actual_price"]
    threshold = float(actual.abs().quantile(0.95))
    overall: dict[str, dict[str, float | None]] = {}
    for model in models:
        overall[model] = {
            "mae_eur_per_mwh": mae(actual, predictions[model]),
            "rmse_eur_per_mwh": rmse(actual, predictions[model]),
            "directional_accuracy_vs_previous_day": (
                None
                if model == "previous_day"
                else directional_accuracy(actual, predictions[model], predictions["previous_day"])
            ),
        }

    error_rows: list[dict[str, object]] = []
    for group_name, group_column in [("hour", "local_hour"), ("month", "month")]:
        for group_value, group in predictions.groupby(group_column):
            for model in models:
                error_rows.append(
                    {
                        "group": group_name,
                        "value": int(str(group_value)),
                        "model": model,
                        "mae_eur_per_mwh": mae(group["actual_price"], group[model]),
                        "observations": len(group),
                    }
                )
    slices: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for label, mask in {
        "negative_price": actual < 0,
        "extreme_absolute_price_top_5pct": actual.abs() >= threshold,
    }.items():
        slices[label] = {}
        for model in models:
            slices[label][model] = {
                "mae_eur_per_mwh": (
                    mae(actual[mask], predictions.loc[mask, model]) if mask.any() else None
                ),
                "observations": int(mask.sum()),
            }

    best_naive = min(
        ["previous_day", "previous_week"],
        key=lambda name: float(overall[name]["mae_eur_per_mwh"] or np.inf),
    )
    bootstrap = {
        model: daily_block_bootstrap_mae_difference(predictions, model, best_naive, samples, seed)
        for model in ["ridge", "hist_gradient_boosting"]
    }
    result: dict[str, object] = {
        "period": {
            "start": str(predictions["local_date"].min()),
            "end": str(predictions["local_date"].max()),
        },
        "unit": "EUR/MWh",
        "directional_metric_definition": (
            "Sign of forecast minus previous-day price equals sign of actual minus "
            "previous-day price; actual ties excluded."
        ),
        "extreme_definition": (
            f"Absolute actual price >= test-set 95th percentile ({threshold:.4f} "
            "EUR/MWh); used for evaluation only."
        ),
        "overall": overall,
        "slices": slices,
        "best_naive_by_validation_independent_test_metric": best_naive,
        "bootstrap_vs_best_naive": bootstrap,
    }
    output = project_root / "artifacts" / "metrics"
    output.mkdir(parents=True, exist_ok=True)
    (output / "forecast_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(error_rows).to_csv(output / "forecast_errors_by_hour_month.csv", index=False)
    return result
