from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from power_market.validation import validate_prices


def _day_frame(settings, target: date) -> tuple[pd.DataFrame, object]:
    data = settings.data.model_copy(update={"start_date": target, "end_date": target})
    scoped = settings.model_copy(update={"data": data})
    start = pd.Timestamp(target, tz="Europe/Berlin").tz_convert("UTC")
    end = (pd.Timestamp(target, tz="Europe/Berlin") + pd.DateOffset(days=1)).tz_convert("UTC")
    timestamps = pd.date_range(start, end, inclusive="left", freq="h")
    return pd.DataFrame(
        {"timestamp_utc": timestamps, "price_eur_per_mwh": np.arange(len(timestamps)) - 5}
    ), scoped


def test_spring_dst_has_23_unique_utc_hours(settings) -> None:
    frame, scoped = _day_frame(settings, date(2024, 3, 31))
    report = validate_prices(frame, scoped)
    assert report["status"] == "pass"
    assert report["summary"]["spring_dst_23_hour_days"] == ["2024-03-31"]
    assert len(frame) == 23


def test_autumn_dst_has_25_hours_and_repeated_local_hour(settings) -> None:
    frame, scoped = _day_frame(settings, date(2024, 10, 27))
    report = validate_prices(frame, scoped)
    local = frame["timestamp_utc"].dt.tz_convert("Europe/Berlin")
    assert report["status"] == "pass"
    assert report["summary"]["autumn_dst_25_hour_days"] == ["2024-10-27"]
    assert (local.dt.hour == 2).sum() == 2
    assert frame["timestamp_utc"].is_unique


def test_missing_duplicate_and_invalid_values_are_reported(settings) -> None:
    frame, scoped = _day_frame(settings, date(2024, 1, 2))
    broken = pd.concat([frame.drop(index=5), frame.iloc[[2]]], ignore_index=True)
    broken.loc[0, "price_eur_per_mwh"] = np.nan
    report = validate_prices(broken, scoped)
    assert report["status"] == "fail"
    assert report["summary"]["missing_observations"] == 1
    assert report["summary"]["duplicates"] == 1
    assert report["summary"]["invalid_numeric_values"] == 1
