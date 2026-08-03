from __future__ import annotations

import pandas as pd

from power_market.data import synthetic_fixture
from power_market.features import (
    FEATURE_COLUMNS,
    assign_split,
    build_features,
    feature_availability_audit,
    german_national_holidays,
)


def test_features_use_exact_past_prices(settings) -> None:
    prices, _ = synthetic_fixture(settings)
    features = build_features(prices, settings)
    assert feature_availability_audit(features)["status"] == "pass"
    assert features.loc[200, "lag_24h_price"] == prices.loc[176, "price_eur_per_mwh"]
    assert features.loc[200, "lag_168h_price"] == prices.loc[32, "price_eur_per_mwh"]
    assert "price_eur_per_mwh" not in FEATURE_COLUMNS


def test_future_target_change_cannot_change_past_features(settings) -> None:
    prices, _ = synthetic_fixture(settings)
    original = build_features(prices.iloc[:500].copy(), settings)
    changed = prices.iloc[:500].copy()
    changed.loc[400:, "price_eur_per_mwh"] += 10000
    rebuilt = build_features(changed, settings)
    pd.testing.assert_frame_equal(
        original.loc[:399, FEATURE_COLUMNS], rebuilt.loc[:399, FEATURE_COLUMNS]
    )


def test_split_boundaries_are_chronological(settings) -> None:
    prices, _ = synthetic_fixture(settings)
    labels = assign_split(prices, settings)
    assert labels.iloc[0] == "train"
    assert (
        labels.loc[prices["timestamp_utc"] >= pd.Timestamp("2024-01-01", tz="UTC")].iloc[0]
        == "validation"
    )
    assert (
        labels.loc[prices["timestamp_utc"] >= pd.Timestamp("2025-01-01", tz="UTC")].iloc[0]
        == "test"
    )
    assert not labels.eq("outside").any()


def test_known_holiday_calendar() -> None:
    holidays = german_national_holidays(2025)
    assert pd.Timestamp("2025-04-18").date() in holidays  # Good Friday
    assert pd.Timestamp("2025-10-03").date() in holidays
    assert len(holidays) == 9
