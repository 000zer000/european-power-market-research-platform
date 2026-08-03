"""Leakage-safe, decision-time-available feature engineering."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from power_market.config import Settings

FEATURE_COLUMNS: Final[list[str]] = [
    "local_hour",
    "weekday",
    "month",
    "is_weekend",
    "is_holiday",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",
    "lag_24h_price",
    "lag_168h_price",
    "rolling_24h_mean_lagged_24h",
    "rolling_168h_mean_lagged_24h",
    "rolling_168h_std_lagged_24h",
]


def _easter_sunday(year: int) -> date:
    """Compute Gregorian Easter using the Meeus/Jones/Butcher algorithm."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month = (h + ell - 7 * m + 114) // 31
    day = (h + ell - 7 * m + 114) % 31 + 1
    return date(year, month, day)


def german_national_holidays(year: int) -> set[date]:
    """Return nationwide German public holidays known before delivery."""
    easter = _easter_sunday(year)
    return {
        date(year, 1, 1),
        easter - timedelta(days=2),
        easter + timedelta(days=1),
        date(year, 5, 1),
        easter + timedelta(days=39),
        easter + timedelta(days=50),
        date(year, 10, 3),
        date(year, 12, 25),
        date(year, 12, 26),
    }


def assign_split(frame: pd.DataFrame, settings: Settings) -> pd.Series:
    """Label rows by local delivery date using immutable chronological boundaries."""
    local_dates = frame["timestamp_utc"].dt.tz_convert(settings.data.market_timezone).dt.date
    labels = np.select(
        [
            local_dates <= settings.splits.train_end,
            local_dates <= settings.splits.validation_end,
            local_dates <= settings.splits.test_end,
        ],
        ["train", "validation", "test"],
        default="outside",
    )
    return pd.Series(labels, index=frame.index, dtype="string")


def build_features(prices: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Build features whose price inputs are lagged at least 24 delivery hours."""
    frame = prices.sort_values("timestamp_utc").reset_index(drop=True).copy()
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    local = frame["timestamp_utc"].dt.tz_convert(settings.data.market_timezone)
    frame["timestamp_local"] = local.astype(str)
    frame["local_date"] = local.dt.date.astype(str)
    frame["local_hour"] = local.dt.hour.astype(int)
    frame["weekday"] = local.dt.weekday.astype(int)
    frame["month"] = local.dt.month.astype(int)
    frame["utc_offset_hours"] = local.map(lambda item: item.utcoffset().total_seconds() / 3600)
    frame["is_weekend"] = (frame["weekday"] >= 5).astype(int)
    holiday_dates = set().union(
        *(
            german_national_holidays(year)
            for year in range(local.dt.year.min(), local.dt.year.max() + 1)
        )
    )
    frame["is_holiday"] = local.dt.date.isin(holiday_dates).astype(int)

    frame["hour_sin"] = np.sin(2 * np.pi * frame["local_hour"] / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * frame["local_hour"] / 24)
    frame["weekday_sin"] = np.sin(2 * np.pi * frame["weekday"] / 7)
    frame["weekday_cos"] = np.cos(2 * np.pi * frame["weekday"] / 7)
    frame["month_sin"] = np.sin(2 * np.pi * (frame["month"] - 1) / 12)
    frame["month_cos"] = np.cos(2 * np.pi * (frame["month"] - 1) / 12)

    prices_series = frame["price_eur_per_mwh"]
    frame["lag_24h_price"] = prices_series.shift(24)
    frame["lag_168h_price"] = prices_series.shift(168)
    safe_history = prices_series.shift(24)
    frame["rolling_24h_mean_lagged_24h"] = safe_history.rolling(24).mean()
    frame["rolling_168h_mean_lagged_24h"] = safe_history.rolling(168).mean()
    frame["rolling_168h_std_lagged_24h"] = safe_history.rolling(168).std()
    frame["split"] = assign_split(frame, settings)
    return frame


def feature_availability_audit(features: pd.DataFrame) -> dict[str, object]:
    """Verify exact lag and rolling-history invariants against the target series."""
    target = features["price_eur_per_mwh"]
    lag_24_matches = np.allclose(
        features["lag_24h_price"].iloc[24:].to_numpy(),
        target.iloc[:-24].to_numpy(),
        equal_nan=True,
    )
    lag_168_matches = np.allclose(
        features["lag_168h_price"].iloc[168:].to_numpy(),
        target.iloc[:-168].to_numpy(),
        equal_nan=True,
    )
    rolling_expected = target.shift(24).rolling(168).mean()
    rolling_matches = np.allclose(
        features["rolling_168h_mean_lagged_24h"], rolling_expected, equal_nan=True
    )
    checks = {
        "lag_24_is_exact_past_value": bool(lag_24_matches),
        "lag_168_is_exact_past_value": bool(lag_168_matches),
        "rolling_window_ends_24h_before_target": bool(rolling_matches),
        "no_target_column_in_model_features": "price_eur_per_mwh" not in FEATURE_COLUMNS,
        "minimum_price_feature_lag_hours": 24,
    }
    return {
        "status": "pass"
        if all(value is True or value == 24 for value in checks.values())
        else "fail",
        "checks": checks,
    }


def save_features(features: pd.DataFrame, settings: Settings) -> tuple[Path, Path]:
    """Persist feature data and a human-auditable availability manifest."""
    processed = settings.project_root / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    path = processed / "features.csv"
    features.to_csv(path, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
    audit = feature_availability_audit(features)
    if audit["status"] != "pass":
        raise ValueError("feature availability audit failed")
    manifest = {
        **audit,
        "forecast_decision": (
            "Forecast the complete delivery day using prices published for earlier "
            "day-ahead auctions."
        ),
        "calendar_features": "Known before the auction.",
        "price_features": (
            "All source observations are at least 24 delivery hours behind the target."
        ),
        "features": FEATURE_COLUMNS,
        "warm_up_rows_with_missing_features": int(
            features[FEATURE_COLUMNS].isna().any(axis=1).sum()
        ),
        "warm_up_policy": (
            "Rows lacking the documented 168-hour history are excluded from model fitting only."
        ),
    }
    manifest_path = settings.project_root / "artifacts" / "provenance" / "feature_availability.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path, manifest_path


def load_features(settings: Settings) -> pd.DataFrame:
    """Load generated features and restore UTC timestamps."""
    path = settings.project_root / "data" / "processed" / "features.csv"
    frame = pd.read_csv(path)
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    return frame
