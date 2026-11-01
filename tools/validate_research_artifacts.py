#!/usr/bin/env python3
"""Verify that committed research outputs remain internally consistent.

This is an evidence-contract check, not a pipeline rerun. It recomputes selected
published statistics from committed tabular outputs and verifies accounting,
coverage, schema, and cross-file invariants.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from power_market.battery import evaluate_schedule
from power_market.config import load_settings

MODELS = ("previous_day", "previous_week", "ridge", "hist_gradient_boosting")
EXPECTED_PREDICTION_COLUMNS = {
    "timestamp_utc",
    "timestamp_local",
    "local_date",
    "local_hour",
    "month",
    "actual_price",
    *MODELS,
}
EXPECTED_STRATEGIES = {
    "do_nothing",
    "time_of_use",
    *MODELS,
    "perfect_foresight",
}


class ContractError(RuntimeError):
    """Raised when committed research evidence violates its contract."""


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{path} must contain a JSON object")
    return value


def require_columns(frame: pd.DataFrame, expected: set[str], path: Path) -> None:
    missing = sorted(expected.difference(frame.columns))
    if missing:
        raise ContractError(f"{path} is missing columns: {', '.join(missing)}")


def require_close(actual: float, expected: float, label: str, tolerance: float = 1e-9) -> None:
    if not math.isfinite(actual) or not math.isfinite(expected):
        raise ContractError(f"{label} must be finite: actual={actual}, expected={expected}")
    if not math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise ContractError(f"{label} mismatch: recomputed={actual}, published={expected}")


def validate_predictions(root: Path, metrics: dict[str, Any]) -> pd.DataFrame:
    path = root / "artifacts/metrics/test_predictions.csv"
    frame = pd.read_csv(path)
    require_columns(frame, EXPECTED_PREDICTION_COLUMNS, path)

    if len(frame) != 8760:
        raise ContractError(f"{path} must contain 8,760 hourly 2025 observations, got {len(frame)}")
    if frame[list(EXPECTED_PREDICTION_COLUMNS)].isna().any().any():
        raise ContractError(f"{path} contains missing values in contract columns")

    utc = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="raise")
    if not utc.is_monotonic_increasing or utc.duplicated().any():
        raise ContractError("prediction UTC timestamps must be unique and strictly ordered")
    spacing = utc.diff().dropna()
    if not (spacing == pd.Timedelta(hours=1)).all():
        raise ContractError("prediction UTC timestamps must form a continuous hourly series")

    if set(frame["local_hour"].astype(int).unique()) != set(range(24)):
        raise ContractError("local_hour must cover every hour from 0 through 23")
    if not frame["month"].between(1, 12).all():
        raise ContractError("month values must lie between 1 and 12")

    overall = metrics.get("overall")
    if not isinstance(overall, dict):
        raise ContractError("forecast_metrics.json requires an 'overall' object")

    actual = frame["actual_price"].astype(float)
    for model in MODELS:
        published = overall.get(model)
        if not isinstance(published, dict):
            raise ContractError(f"forecast_metrics.json is missing overall metrics for {model}")
        predicted = frame[model].astype(float)
        mae = float((predicted - actual).abs().mean())
        rmse = float(((predicted - actual) ** 2).mean() ** 0.5)
        require_close(mae, float(published["mae_eur_per_mwh"]), f"{model} MAE")
        require_close(rmse, float(published["rmse_eur_per_mwh"]), f"{model} RMSE")

    period = metrics.get("period")
    if period != {"start": "2025-01-01", "end": "2025-12-31"}:
        raise ContractError(f"unexpected published test period: {period!r}")
    if metrics.get("unit") != "EUR/MWh":
        raise ContractError("forecast metric unit must be EUR/MWh")
    return frame


def validate_battery_results(root: Path, predictions: pd.DataFrame) -> None:
    path = root / "artifacts/backtests/battery_daily_results.csv"
    frame = pd.read_csv(path)
    required = {
        "local_date",
        "strategy",
        "intervals",
        "realised_gross_market_revenue_eur",
        "degradation_cost_eur",
        "realised_net_operating_value_eur",
        "internal_throughput_mwh",
        "equivalent_full_cycles",
        "perfect_foresight_gap_eur",
    }
    require_columns(frame, required, path)

    observed_strategies = set(frame["strategy"].astype(str).unique())
    if observed_strategies != EXPECTED_STRATEGIES:
        raise ContractError(
            f"unexpected strategy set: expected {sorted(EXPECTED_STRATEGIES)}, "
            f"got {sorted(observed_strategies)}"
        )

    counts = frame.groupby("strategy", observed=True)["local_date"].nunique()
    if not (counts == 365).all():
        raise ContractError(f"each strategy must cover 365 local days; got {counts.to_dict()}")
    if len(frame) != 365 * len(EXPECTED_STRATEGIES):
        raise ContractError(f"unexpected battery result row count: {len(frame)}")

    gross = frame["realised_gross_market_revenue_eur"].astype(float)
    degradation = frame["degradation_cost_eur"].astype(float)
    net = frame["realised_net_operating_value_eur"].astype(float)
    max_accounting_error = float((net - (gross - degradation)).abs().max())
    if max_accounting_error > 1e-9:
        raise ContractError(f"battery net-value accounting error is {max_accounting_error}")

    if (frame["degradation_cost_eur"].astype(float) < -1e-12).any():
        raise ContractError("degradation cost cannot be negative")
    if (frame["internal_throughput_mwh"].astype(float) < -1e-12).any():
        raise ContractError("internal throughput cannot be negative")
    if (frame["equivalent_full_cycles"].astype(float) < -1e-12).any():
        raise ContractError("equivalent full cycles cannot be negative")
    if (frame["perfect_foresight_gap_eur"].astype(float) < -1e-9).any():
        raise ContractError("perfect-foresight gap cannot be materially negative")

    perfect = frame.loc[frame["strategy"] == "perfect_foresight", "perfect_foresight_gap_eur"]
    if float(perfect.astype(float).abs().max()) > 1e-9:
        raise ContractError("perfect-foresight strategy must have zero perfect-foresight gap")

    prediction_days = set(predictions["local_date"].astype(str).unique())
    battery_days = set(frame["local_date"].astype(str).unique())
    if prediction_days != battery_days:
        raise ContractError("prediction and battery-result local-date coverage differ")

    # Audit interval physics independently, then reconcile every daily metric.
    dispatch = pd.read_csv(root / "artifacts/backtests/battery_dispatch.csv")
    require_columns(
        dispatch, {"local_date", "strategy", "timestamp", "actual_price_eur_per_mwh"}, path
    )
    battery = load_settings(root / "configs/default.yaml").battery
    daily = frame.set_index(["local_date", "strategy"], verify_integrity=True)
    actual_by_stamp = predictions.set_index("timestamp_utc")["actual_price"]
    actual_by_stamp.index = pd.to_datetime(actual_by_stamp.index, utc=True)
    groups = dispatch.groupby(["local_date", "strategy"], sort=False)
    if set(groups.groups) != set(daily.index):
        raise ContractError("dispatch and daily result horizons differ")
    for key, schedule in groups:
        stamps = pd.DatetimeIndex(pd.to_datetime(schedule["timestamp"], utc=True))
        actual = actual_by_stamp.reindex(stamps)
        if actual.isna().any():
            raise ContractError(f"dispatch contains an unknown prediction timestamp: {key}")
        published_actual = schedule["actual_price_eur_per_mwh"].to_numpy(dtype=float)
        if not (published_actual == actual.to_numpy()).all():
            raise ContractError(f"dispatch settlement prices differ from predictions: {key}")
        try:
            values = evaluate_schedule(schedule, actual, battery)
        except (ValueError, RuntimeError) as exc:
            raise ContractError(f"invalid dispatch {key}: {exc}") from exc
        require_close(float(len(schedule)), float(daily.loc[key, "intervals"]), f"{key} intervals")
        for metric, value in values.items():
            require_close(value, float(daily.loc[key, metric]), f"{key} {metric}")


def validate_forecast_to_value(root: Path, predictions: pd.DataFrame) -> None:
    published = load_json(root / "artifacts/backtests/forecast_to_value.json")
    battery = pd.read_csv(root / "artifacts/backtests/battery_daily_results.csv")
    actual = predictions["actual_price"].astype(float)
    dated = predictions.assign(_actual=actual)

    for model in MODELS:
        daily_mae = (
            dated.assign(_error=(dated[model].astype(float) - dated["_actual"]).abs())
            .groupby("local_date", observed=True)["_error"]
            .mean()
        )
        gaps = (
            battery.loc[battery["strategy"] == model]
            .set_index("local_date")["perfect_foresight_gap_eur"]
            .astype(float)
        )
        aligned = pd.concat([daily_mae.rename("mae"), gaps.rename("gap")], axis=1).dropna()
        correlation = float(aligned["mae"].corr(aligned["gap"]))
        model_published = published.get(model)
        if not isinstance(model_published, dict):
            raise ContractError(f"forecast_to_value.json is missing {model}")
        require_close(
            correlation,
            float(model_published["pearson_daily_mae_vs_perfect_value_gap"]),
            f"{model} forecast-to-value correlation",
        )


def validate(root: Path) -> None:
    required_paths = [
        "artifacts/metrics/forecast_metrics.json",
        "artifacts/metrics/test_predictions.csv",
        "artifacts/backtests/battery_daily_results.csv",
        "artifacts/backtests/forecast_to_value.json",
        "artifacts/validation/data_validation.json",
        "artifacts/provenance/manifest.json",
    ]
    missing = [relative for relative in required_paths if not (root / relative).is_file()]
    if missing:
        raise ContractError(f"missing committed evidence files: {', '.join(missing)}")

    metrics = load_json(root / "artifacts/metrics/forecast_metrics.json")
    predictions = validate_predictions(root, metrics)
    validate_battery_results(root, predictions)
    validate_forecast_to_value(root, predictions)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    try:
        validate(args.root.resolve())
    except (ContractError, OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"research artifact contract failed: {exc}\n")
    print("research artifact contract passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
