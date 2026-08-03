"""Explicit data-quality checks with no silent repair."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from power_market.config import Settings


class DataValidationError(ValueError):
    """Raised when canonical market data fail a required invariant."""


def validate_prices(frame: pd.DataFrame, settings: Settings) -> dict[str, Any]:
    """Validate schema, UTC continuity, values and local-market DST day lengths."""
    required = {"timestamp_utc", "price_eur_per_mwh"}
    missing_columns = sorted(required - set(frame.columns))
    timestamp_source = (
        frame["timestamp_utc"] if "timestamp_utc" in frame else pd.Series([pd.NaT] * len(frame))
    )
    price_source = (
        frame["price_eur_per_mwh"]
        if "price_eur_per_mwh" in frame
        else pd.Series([np.nan] * len(frame))
    )
    timestamps = pd.to_datetime(timestamp_source, utc=True, errors="coerce")
    prices = pd.to_numeric(price_source, errors="coerce")
    duplicate_count = int(timestamps.duplicated().sum())
    invalid_timestamp_count = int(timestamps.isna().sum())
    invalid_numeric_count = int((~np.isfinite(prices)).sum()) if not prices.empty else len(frame)
    monotonic = bool(timestamps.is_monotonic_increasing)

    timezone = ZoneInfo(settings.data.market_timezone)
    expected_start = pd.Timestamp(
        datetime.combine(settings.data.start_date, datetime.min.time(), timezone)
    ).tz_convert("UTC")
    expected_end = pd.Timestamp(
        datetime.combine(settings.data.end_date + timedelta(days=1), datetime.min.time(), timezone)
    ).tz_convert("UTC")
    expected = pd.date_range(expected_start, expected_end, inclusive="left", freq="h")
    observed_index = pd.DatetimeIndex(timestamps.dropna())
    missing_observations = expected.difference(observed_index)
    unexpected_observations = observed_index.difference(expected)

    local_dates = timestamps.dt.tz_convert(timezone).dt.date
    day_counts = local_dates.value_counts().sort_index()
    spring_days = [str(day) for day, count in day_counts.items() if count == 23]
    autumn_days = [str(day) for day, count in day_counts.items() if count == 25]
    invalid_day_counts = {
        str(day): int(count) for day, count in day_counts.items() if count not in {23, 24, 25}
    }

    checks = {
        "required_columns_present": not missing_columns,
        "timestamps_valid": invalid_timestamp_count == 0,
        "timestamps_unique": duplicate_count == 0,
        "timestamps_monotonic": monotonic,
        "numeric_values_finite": invalid_numeric_count == 0,
        "no_missing_hourly_utc_observations": len(missing_observations) == 0,
        "no_unexpected_observations": len(unexpected_observations) == 0,
        "local_day_lengths_valid": not invalid_day_counts,
    }
    report: dict[str, Any] = {
        "status": "pass" if all(checks.values()) else "fail",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "checks": checks,
        "summary": {
            "rows": len(frame),
            "first_timestamp_utc": timestamps.min().isoformat() if len(frame) else None,
            "last_timestamp_utc": timestamps.max().isoformat() if len(frame) else None,
            "negative_price_observations": int((prices < 0).sum()),
            "zero_price_observations": int((prices == 0).sum()),
            "minimum_price_eur_per_mwh": float(prices.min()) if len(frame) else None,
            "maximum_price_eur_per_mwh": float(prices.max()) if len(frame) else None,
            "missing_observations": len(missing_observations),
            "unexpected_observations": len(unexpected_observations),
            "duplicates": duplicate_count,
            "invalid_numeric_values": invalid_numeric_count,
            "spring_dst_23_hour_days": spring_days,
            "autumn_dst_25_hour_days": autumn_days,
            "invalid_local_day_counts": invalid_day_counts,
        },
        "issues": {
            "missing_columns": missing_columns,
            "missing_timestamp_examples": [item.isoformat() for item in missing_observations[:10]],
            "unexpected_timestamp_examples": [
                item.isoformat() for item in unexpected_observations[:10]
            ],
        },
        "policy": "No values were interpolated, dropped, clipped, deduplicated or imputed.",
    }
    return report


def write_validation_report(report: dict[str, Any], project_root: Path) -> Path:
    """Persist the machine-readable validation evidence."""
    path = project_root / "artifacts" / "validation" / "data_validation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def require_valid_prices(frame: pd.DataFrame, settings: Settings) -> dict[str, Any]:
    """Write a validation report and stop the pipeline on any failure."""
    report = validate_prices(frame, settings)
    write_validation_report(report, settings.project_root)
    if report["status"] != "pass":
        failed = [name for name, passed in report["checks"].items() if not passed]
        raise DataValidationError(f"price validation failed: {', '.join(failed)}")
    return report
