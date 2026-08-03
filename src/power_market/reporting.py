"""Publication-quality static figures and generated research summary."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from power_market.config import Settings

COLORS = {
    "previous_day": "#7F8C8D",
    "previous_week": "#BDC3C7",
    "ridge": "#2E86AB",
    "hist_gradient_boosting": "#F18F01",
    "time_of_use": "#7D3C98",
    "perfect_foresight": "#2A9D8F",
    "do_nothing": "#1F2937",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "font.size": 10,
            "axes.titleweight": "bold",
        }
    )


def _save(fig: Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _price_distribution(prices: pd.DataFrame, path: Path, timezone: str) -> None:
    local = prices.assign(local=prices["timestamp_utc"].dt.tz_convert(timezone))
    local["year"] = local["local"].dt.year
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    clipped = local["price_eur_per_mwh"].clip(
        local["price_eur_per_mwh"].quantile(0.005), local["price_eur_per_mwh"].quantile(0.995)
    )
    axes[0].hist(clipped, bins=70, color="#2E86AB", alpha=0.9)
    axes[0].axvline(0, color="#C0392B", linewidth=1.2)
    axes[0].set(
        title="DE-LU day-ahead price distribution",
        xlabel="Price (EUR/MWh)",
        ylabel="Hourly observations",
    )
    negative = (
        local.assign(negative=local["price_eur_per_mwh"] < 0).groupby("year")["negative"].mean()
        * 100
    )
    axes[1].bar(negative.index.astype(str), negative, color="#C0392B")
    axes[1].set(
        title="Negative-price frequency", xlabel="Delivery year", ylabel="Share of hours (%)"
    )
    _save(fig, path)


def _forecast_example(predictions: pd.DataFrame, path: Path) -> None:
    example = predictions.iloc[: 24 * 7]
    fig, axis = plt.subplots(figsize=(12, 4.5))
    axis.plot(
        example["timestamp_utc"],
        example["actual_price"],
        color="#111827",
        linewidth=2,
        label="Actual",
    )
    for model in ["previous_day", "ridge", "hist_gradient_boosting"]:
        axis.plot(
            example["timestamp_utc"],
            example[model],
            color=COLORS[model],
            linewidth=1.2,
            label=model.replace("_", " ").title(),
        )
    axis.set(
        title="One-week out-of-sample forecast example",
        xlabel="Delivery time (UTC)",
        ylabel="Price (EUR/MWh)",
    )
    axis.xaxis.set_major_formatter(
        mdates.DateFormatter("%d %b")  # type: ignore[no-untyped-call]
    )
    axis.legend(ncol=4, frameon=False)
    _save(fig, path)


def _model_errors(metrics: dict[str, Any], path: Path) -> None:
    models = list(metrics["overall"])
    mae_values = [metrics["overall"][model]["mae_eur_per_mwh"] for model in models]
    rmse_values = [metrics["overall"][model]["rmse_eur_per_mwh"] for model in models]
    x = np.arange(len(models))
    fig, axis = plt.subplots(figsize=(9, 4.5))
    axis.bar(x - 0.18, mae_values, 0.36, label="MAE", color="#2E86AB")
    axis.bar(x + 0.18, rmse_values, 0.36, label="RMSE", color="#F18F01")
    axis.set_xticks(
        x, [name.replace("hist_gradient_boosting", "HGBR").replace("_", "\n") for name in models]
    )
    axis.set(title="Final-test forecast error", ylabel="Error (EUR/MWh)")
    axis.legend(frameon=False)
    _save(fig, path)


def _hourly_errors(project_root: Path, path: Path) -> None:
    errors = pd.read_csv(
        project_root / "artifacts" / "metrics" / "forecast_errors_by_hour_month.csv"
    )
    errors = errors.loc[errors["group"].eq("hour")]
    fig, axis = plt.subplots(figsize=(10, 4.5))
    for model_value, group in errors.groupby("model"):
        model = str(model_value)
        axis.plot(
            group["value"],
            group["mae_eur_per_mwh"],
            marker="o",
            markersize=3,
            label=model.replace("_", " ").title(),
            color=COLORS[model],
        )
    axis.set(
        title="MAE by local delivery hour",
        xlabel="Europe/Berlin hour",
        ylabel="MAE (EUR/MWh)",
        xticks=range(0, 24, 2),
    )
    axis.legend(frameon=False, ncol=2)
    _save(fig, path)


def _dispatch_example(intervals: pd.DataFrame, aggregate: pd.DataFrame, path: Path) -> None:
    candidates = aggregate.loc[aggregate["strategy"].isin(["ridge", "hist_gradient_boosting"])]
    strategy = candidates.sort_values("realised_net_operating_value_eur", ascending=False)[
        "strategy"
    ].iloc[0]
    day_values = (
        intervals.loc[intervals["strategy"].eq(strategy)]
        .groupby("local_date")["actual_price_eur_per_mwh"]
        .min()
    )
    chosen_date = day_values.idxmin()
    example = intervals.loc[
        (intervals["strategy"].eq(strategy)) & (intervals["local_date"].eq(chosen_date))
    ]
    fig, (price_axis, dispatch_axis) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    price_axis.plot(
        example["timestamp"],
        example["actual_price_eur_per_mwh"],
        color="#111827",
        label="Actual price",
    )
    price_axis.plot(
        example["timestamp"],
        example["decision_price_eur_per_mwh"],
        color=COLORS[strategy],
        label="Forecast price",
    )
    price_axis.set(
        ylabel="Price (EUR/MWh)",
        title=f"Forecast-driven dispatch example — {chosen_date} ({strategy.replace('_', ' ')})",
    )
    price_axis.legend(frameon=False, ncol=2)
    net_dispatch = example["discharge_mw"] - example["charge_mw"]
    dispatch_axis.bar(
        example["timestamp"],
        net_dispatch,
        width=0.025,
        color=np.where(net_dispatch >= 0, "#2A9D8F", "#C0392B"),
        label="Net dispatch",
    )
    soc_axis = dispatch_axis.twinx()
    soc_axis.step(
        example["timestamp"],
        example["soc_end_mwh"],
        where="post",
        color="#2E86AB",
        linewidth=2,
        label="State of charge",
    )
    dispatch_axis.set(ylabel="Power (MW)", xlabel="Delivery time (UTC)")
    soc_axis.set_ylabel("State of charge (MWh)")
    _save(fig, path)


def _cumulative_value(daily: pd.DataFrame, path: Path) -> None:
    pivot = daily.pivot(
        index="local_date", columns="strategy", values="realised_net_operating_value_eur"
    ).cumsum()
    fig, axis = plt.subplots(figsize=(11, 4.8))
    for strategy in [
        "time_of_use",
        *["previous_day", "previous_week", "ridge", "hist_gradient_boosting"],
        "perfect_foresight",
    ]:
        axis.plot(
            pd.to_datetime(pivot.index),
            pivot[strategy],
            label=strategy.replace("_", " ").title(),
            color=COLORS[strategy],
        )
    axis.set(
        title="Cumulative realised net operating value",
        xlabel="2025 delivery date",
        ylabel="Value (EUR)",
    )
    axis.legend(frameon=False, ncol=3)
    _save(fig, path)


def _perfect_gap(aggregate: pd.DataFrame, path: Path) -> None:
    data = aggregate.loc[
        ~aggregate["strategy"].isin(["perfect_foresight", "do_nothing"])
    ].sort_values("perfect_foresight_gap_eur")
    fig, axis = plt.subplots(figsize=(9, 4.5))
    axis.barh(
        data["strategy"].str.replace("_", " ").str.title(),
        data["perfect_foresight_gap_eur"],
        color=[COLORS[name] for name in data["strategy"]],
    )
    axis.set(
        title="Gap to perfect-foresight upper bound",
        xlabel="Foregone modeled net operating value (EUR)",
    )
    _save(fig, path)


def _sensitivity(sensitivity: pd.DataFrame, path: Path) -> None:
    data = sensitivity.sort_values("perfect_foresight_net_value_eur")
    fig, axis = plt.subplots(figsize=(10, 5))
    colors = ["#2E86AB" if name == "base" else "#7F8C8D" for name in data["scenario"]]
    axis.barh(
        data["scenario"].str.replace("_", " "),
        data["perfect_foresight_net_value_eur"],
        color=colors,
    )
    axis.set(
        title="Battery assumption sensitivity (perfect foresight)",
        xlabel="2025 modeled net operating value (EUR)",
    )
    _save(fig, path)


def generate_figures(
    prices: pd.DataFrame,
    predictions: pd.DataFrame,
    metrics: dict[str, Any],
    daily: pd.DataFrame,
    aggregate: pd.DataFrame,
    intervals: pd.DataFrame,
    sensitivity: pd.DataFrame,
    settings: Settings,
) -> list[Path]:
    """Generate all recruiter-facing evidence figures."""
    _style()
    output = settings.project_root / "artifacts" / "figures"
    output.mkdir(parents=True, exist_ok=True)
    tasks: list[tuple[str, Callable[[Path], None]]] = [
        (
            "price_distribution.png",
            lambda path: _price_distribution(prices, path, settings.data.market_timezone),
        ),
        ("forecast_example.png", lambda path: _forecast_example(predictions, path)),
        ("model_error_comparison.png", lambda path: _model_errors(metrics, path)),
        ("error_by_hour.png", lambda path: _hourly_errors(settings.project_root, path)),
        (
            "battery_dispatch_example.png",
            lambda path: _dispatch_example(intervals, aggregate, path),
        ),
        ("cumulative_realised_value.png", lambda path: _cumulative_value(daily, path)),
        ("perfect_foresight_gap.png", lambda path: _perfect_gap(aggregate, path)),
        ("battery_sensitivity.png", lambda path: _sensitivity(sensitivity, path)),
    ]
    paths: list[Path] = []
    for filename, function in tasks:
        path = output / filename
        function(path)
        paths.append(path)
    return paths


def generate_summary(metrics: dict[str, Any], aggregate: pd.DataFrame, settings: Settings) -> Path:
    """Write a concise summary containing only executed measurements."""
    overall = cast(dict[str, dict[str, float]], metrics["overall"])
    ranking = sorted(overall.items(), key=lambda item: item[1]["mae_eur_per_mwh"])
    battery = aggregate.set_index("strategy")
    best_forecast_strategy = str(
        battery.loc[["previous_day", "previous_week", "ridge", "hist_gradient_boosting"]][
            "realised_net_operating_value_eur"
        ].idxmax()
    )
    best_value = float(str(battery.loc[best_forecast_strategy, "realised_net_operating_value_eur"]))
    available_days = int(str(battery.loc[best_forecast_strategy, "available_test_days"]))
    perfect_value = float(str(battery.loc["perfect_foresight", "realised_net_operating_value_eur"]))
    source_line = (
        f"Data source: **{settings.data.source}**, DE-LU day-ahead prices, "
        f"{settings.data.start_date} through {settings.data.end_date}. The final 2025 "
        "delivery year was not used for hyperparameter selection."
    )
    forecast_line = (
        f"The lowest final-test MAE was **{ranking[0][0].replace('_', ' ')}** at "
        f"**{ranking[0][1]['mae_eur_per_mwh']:.3f} EUR/MWh** (RMSE "
        f"**{ranking[0][1]['rmse_eur_per_mwh']:.3f} EUR/MWh**). This is an "
        "evaluation result, not a claim that machine learning must outperform "
        "seasonal baselines."
    )
    battery_line = (
        f"Among forecast-driven strategies, **{best_forecast_strategy.replace('_', ' ')}** "
        "produced the highest realised modeled net operating value: "
        f"**EUR {best_value:.2f}** over **{available_days}** days. "
        "Perfect foresight produced "
        f"**EUR {perfect_value:.2f}**, "
        "an upper bound under the same physical and cost assumptions."
    )
    caveat = (
        "“Net operating value” is realised day-ahead market revenue minus the configured "
        "internal-throughput degradation charge. It is not accounting profit: capex, "
        "market-access fees, imbalance costs, bidding restrictions, taxes and operational "
        "risk are excluded."
    )
    text = (
        "# Executed research summary\n\n"
        f"{source_line}\n\n"
        "## Forecasting\n\n"
        f"{forecast_line}\n\n"
        "## Battery backtest\n\n"
        f"{battery_line}\n\n"
        f"{caveat}\n"
    )
    path = settings.project_root / "reports" / "research_summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def load_metrics(project_root: Path) -> dict[str, Any]:
    """Load machine-readable forecast metrics."""
    payload = json.loads(
        (project_root / "artifacts" / "metrics" / "forecast_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    return cast(dict[str, Any], payload)
