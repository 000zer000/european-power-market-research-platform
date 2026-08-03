"""Composable pipeline stages used by the CLI and integration tests."""

from __future__ import annotations

from typing import Any

import pandas as pd

from power_market.backtest import load_backtest, run_backtest, run_sensitivity
from power_market.config import Settings
from power_market.data import fetch_prices, load_prices
from power_market.features import build_features, load_features, save_features
from power_market.forecasting import load_predictions, train_and_predict
from power_market.metrics import evaluate_forecasts
from power_market.reporting import generate_figures, generate_summary, load_metrics
from power_market.validation import require_valid_prices


def fetch_stage(settings: Settings, force: bool = False) -> pd.DataFrame:
    """Fetch and cache source data."""
    frame, _ = fetch_prices(settings, force)
    return frame


def validate_stage(settings: Settings) -> dict[str, Any]:
    """Validate canonical source data."""
    return require_valid_prices(load_prices(settings), settings)


def feature_stage(settings: Settings) -> pd.DataFrame:
    """Build and audit model features."""
    features = build_features(load_prices(settings), settings)
    save_features(features, settings)
    return features


def train_stage(settings: Settings) -> pd.DataFrame:
    """Select models and generate walk-forward final-test forecasts."""
    predictions, _ = train_and_predict(load_features(settings), settings)
    return predictions


def evaluate_stage(settings: Settings) -> dict[str, object]:
    """Calculate forecast metrics and uncertainty."""
    predictions = load_predictions(settings.project_root)
    return evaluate_forecasts(
        predictions,
        settings.forecast.bootstrap_samples,
        settings.random_seed,
        settings.project_root,
    )


def optimise_stage(settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run daily strategy backtests and battery sensitivity."""
    predictions = load_predictions(settings.project_root)
    daily, aggregate, intervals = run_backtest(predictions, settings)
    run_sensitivity(predictions, settings)
    return daily, aggregate, intervals


def report_stage(settings: Settings) -> list[str]:
    """Generate figures and the concise research summary from persisted evidence."""
    prices = load_prices(settings)
    predictions = load_predictions(settings.project_root)
    metrics = load_metrics(settings.project_root)
    daily, aggregate, intervals = load_backtest(settings.project_root)
    sensitivity = pd.read_csv(
        settings.project_root / "artifacts" / "backtests" / "battery_sensitivity.csv"
    )
    figures = generate_figures(
        prices, predictions, metrics, daily, aggregate, intervals, sensitivity, settings
    )
    summary = generate_summary(metrics, aggregate, settings)
    return [str(path) for path in [*figures, summary]]


def run_all(settings: Settings, force: bool = False) -> dict[str, Any]:
    """Reproduce the complete analysis from source to inspected artifacts."""
    prices = fetch_stage(settings, force)
    validation = validate_stage(settings)
    feature_stage(settings)
    predictions = train_stage(settings)
    evaluate_stage(settings)
    optimise_stage(settings)
    outputs = report_stage(settings)
    figures, summary = outputs[:-1], outputs[-1]
    return {
        "validation": validation["status"],
        "observations": len(prices),
        "test_predictions": len(predictions),
        "figures": figures,
        "summary": summary,
    }
