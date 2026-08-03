"""Daily forecast-to-schedule-to-realised-value battery backtest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from power_market.battery import evaluate_schedule, idle_schedule, optimise_schedule
from power_market.config import BatteryConfig, Settings

FORECAST_STRATEGIES = ["previous_day", "previous_week", "ridge", "hist_gradient_boosting"]


def time_of_use_prices(local_hours: pd.Series) -> np.ndarray:
    """Return a fixed, known-in-advance off-peak/peak dispatch signal."""
    hours = local_hours.to_numpy()
    return np.where(
        (hours >= 0) & (hours <= 5), 0.0, np.where((hours >= 17) & (hours <= 20), 100.0, 50.0)
    )


def _run_day(
    day: pd.DataFrame, battery: BatteryConfig
) -> tuple[list[dict[str, Any]], list[pd.DataFrame]]:
    actual = day["actual_price"].to_numpy()
    timestamps = day["timestamp_utc"].tolist()
    decision_vectors = {
        "time_of_use": time_of_use_prices(day["local_hour"]),
        **{strategy: day[strategy].to_numpy() for strategy in FORECAST_STRATEGIES},
        "perfect_foresight": actual,
    }
    schedules = {"do_nothing": idle_schedule(len(day), battery, timestamps)}
    for strategy, decision_prices in decision_vectors.items():
        schedules[strategy] = optimise_schedule(
            decision_prices, battery, timestamps=timestamps
        ).schedule
    daily_rows: list[dict[str, Any]] = []
    interval_frames: list[pd.DataFrame] = []
    for strategy, schedule in schedules.items():
        values = evaluate_schedule(schedule, actual, battery)
        daily_rows.append(
            {
                "local_date": day["local_date"].iloc[0],
                "strategy": strategy,
                "intervals": len(day),
                "contains_negative_price": bool((actual < 0).any()),
                **values,
            }
        )
        interval = schedule.copy()
        interval["strategy"] = strategy
        interval["local_date"] = day["local_date"].iloc[0]
        interval["actual_price_eur_per_mwh"] = actual
        interval_frames.append(interval)
    perfect_value = next(
        row["realised_net_operating_value_eur"]
        for row in daily_rows
        if row["strategy"] == "perfect_foresight"
    )
    for row in daily_rows:
        row["perfect_foresight_gap_eur"] = perfect_value - row["realised_net_operating_value_eur"]
        if row["perfect_foresight_gap_eur"] < -1e-5:
            raise ValueError("perfect foresight failed to upper-bound a feasible strategy")
    return daily_rows, interval_frames


def aggregate_backtest(daily: pd.DataFrame) -> pd.DataFrame:
    """Aggregate realised value while keeping gross revenue and costs separate."""
    rows: list[dict[str, Any]] = []
    for strategy, group in daily.groupby("strategy"):
        negative_days = group.loc[group["contains_negative_price"]]
        rows.append(
            {
                "strategy": strategy,
                "available_test_days": len(group),
                "realised_gross_market_revenue_eur": group[
                    "realised_gross_market_revenue_eur"
                ].sum(),
                "degradation_cost_eur": group["degradation_cost_eur"].sum(),
                "realised_net_operating_value_eur": group["realised_net_operating_value_eur"].sum(),
                "net_value_eur_per_available_day": group["realised_net_operating_value_eur"].mean(),
                "internal_throughput_mwh": group["internal_throughput_mwh"].sum(),
                "equivalent_full_cycles": group["equivalent_full_cycles"].sum(),
                "perfect_foresight_gap_eur": group["perfect_foresight_gap_eur"].sum(),
                "negative_price_days": len(negative_days),
                "net_value_on_negative_price_days_eur": negative_days[
                    "realised_net_operating_value_eur"
                ].sum(),
            }
        )
    return pd.DataFrame(rows).sort_values("realised_net_operating_value_eur", ascending=False)


def _forecast_value_relationship(predictions: pd.DataFrame, daily: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    perfect = daily.loc[daily["strategy"].eq("perfect_foresight")].set_index("local_date")
    for strategy in FORECAST_STRATEGIES:
        error = (
            predictions.assign(
                abs_error=(predictions["actual_price"] - predictions[strategy]).abs()
            )
            .groupby("local_date")["abs_error"]
            .mean()
        )
        value = daily.loc[daily["strategy"].eq(strategy)].set_index("local_date")
        aligned = pd.concat(
            [
                error.rename("daily_mae"),
                (
                    perfect["realised_net_operating_value_eur"]
                    - value["realised_net_operating_value_eur"]
                ).rename("value_gap"),
            ],
            axis=1,
        ).dropna()
        correlation = np.asarray(aligned.corr(), dtype=float)[0, 1]
        result[strategy] = {
            "pearson_daily_mae_vs_perfect_value_gap": float(correlation),
            "interpretation": (
                "Positive means larger price error tends to accompany a larger value gap; "
                "correlation is not causation."
            ),
        }
    return result


def run_backtest(
    predictions: pd.DataFrame, settings: Settings
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run every strategy on each complete local-market test day."""
    daily_rows: list[dict[str, Any]] = []
    interval_frames: list[pd.DataFrame] = []
    for _, day in predictions.groupby("local_date", sort=True):
        rows, day_intervals = _run_day(day, settings.battery)
        daily_rows.extend(rows)
        interval_frames.extend(day_intervals)
    daily = pd.DataFrame(daily_rows)
    interval_output = pd.concat(interval_frames, ignore_index=True)
    aggregate = aggregate_backtest(daily)
    output = settings.project_root / "artifacts" / "backtests"
    output.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output / "battery_daily_results.csv", index=False)
    interval_output.to_csv(output / "battery_dispatch.csv", index=False)
    aggregate.to_csv(output / "battery_summary.csv", index=False)
    relationship = _forecast_value_relationship(predictions, daily)
    (output / "forecast_to_value.json").write_text(
        json.dumps(relationship, indent=2), encoding="utf-8"
    )
    return daily, aggregate, interval_output


def _sensitivity_configs(base: BatteryConfig) -> list[tuple[str, BatteryConfig]]:
    configs: list[tuple[str, BatteryConfig]] = [("base", base)]
    for efficiency in [0.85, 0.90, 0.98]:
        configs.append(
            (
                f"efficiency_{efficiency:.2f}",
                base.model_copy(
                    update={
                        "charge_efficiency": efficiency,
                        "discharge_efficiency": efficiency,
                    }
                ),
            )
        )
    for ratio in [0.25, 1.0]:
        power = base.energy_capacity_mwh * ratio
        configs.append(
            (
                f"power_capacity_ratio_{ratio:.2f}",
                base.model_copy(
                    update={
                        "max_charge_power_mw": power,
                        "max_discharge_power_mw": power,
                    }
                ),
            )
        )
    for cost in [0.0, 10.0]:
        configs.append(
            (
                f"degradation_{cost:.0f}",
                base.model_copy(update={"degradation_cost_eur_per_mwh": cost}),
            )
        )
    return configs


def run_sensitivity(predictions: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Isolate physical/economic assumptions using the perfect-foresight upper bound."""
    rows: list[dict[str, Any]] = []
    for name, battery in _sensitivity_configs(settings.battery):
        total = {"net": 0.0, "gross": 0.0, "cost": 0.0, "cycles": 0.0, "days": 0}
        for _, day in predictions.groupby("local_date", sort=True):
            actual = day["actual_price"].to_numpy()
            schedule = optimise_schedule(actual, battery, timestamps=day["timestamp_utc"]).schedule
            value = evaluate_schedule(schedule, actual, battery)
            total["net"] += value["realised_net_operating_value_eur"]
            total["gross"] += value["realised_gross_market_revenue_eur"]
            total["cost"] += value["degradation_cost_eur"]
            total["cycles"] += value["equivalent_full_cycles"]
            total["days"] += 1
        rows.append(
            {
                "scenario": name,
                "charge_efficiency": battery.charge_efficiency,
                "discharge_efficiency": battery.discharge_efficiency,
                "power_capacity_ratio_per_hour": battery.max_discharge_power_mw
                / battery.energy_capacity_mwh,
                "degradation_cost_eur_per_mwh": battery.degradation_cost_eur_per_mwh,
                "perfect_foresight_gross_revenue_eur": total["gross"],
                "degradation_cost_eur": total["cost"],
                "perfect_foresight_net_value_eur": total["net"],
                "equivalent_full_cycles": total["cycles"],
                "test_days": total["days"],
            }
        )
    frame = pd.DataFrame(rows)
    output = settings.project_root / "artifacts" / "backtests" / "battery_sensitivity.csv"
    frame.to_csv(output, index=False)
    return frame


def load_backtest(project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load persisted daily, aggregate and interval backtest outputs."""
    root = project_root / "artifacts" / "backtests"
    daily = pd.read_csv(root / "battery_daily_results.csv")
    aggregate = pd.read_csv(root / "battery_summary.csv")
    intervals = pd.read_csv(root / "battery_dispatch.csv")
    intervals["timestamp"] = pd.to_datetime(intervals["timestamp"], utc=True)
    return daily, aggregate, intervals
