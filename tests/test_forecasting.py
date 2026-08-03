from __future__ import annotations

import pandas as pd

from power_market.data import synthetic_fixture
from power_market.features import build_features
from power_market.forecasting import select_models, walk_forward_predict


def test_selection_and_predictions_are_deterministic(compact_settings) -> None:
    prices, _ = synthetic_fixture(compact_settings)
    features = build_features(prices, compact_settings)
    selected, leaderboard = select_models(features, compact_settings)
    first = walk_forward_predict(features, selected, compact_settings)
    second = walk_forward_predict(features, selected, compact_settings)
    pd.testing.assert_frame_equal(first, second)
    assert set(leaderboard["model"]) == {"ridge", "hist_gradient_boosting"}
    assert first["local_date"].min() == "2023-03-01"
    assert first["local_date"].max() == "2023-03-07"


def test_test_targets_cannot_affect_validation_selection(compact_settings) -> None:
    prices, _ = synthetic_fixture(compact_settings)
    features = build_features(prices, compact_settings)
    original, table = select_models(features, compact_settings)
    changed = features.copy()
    changed.loc[changed["split"].eq("test"), "price_eur_per_mwh"] += 10000
    rebuilt, changed_table = select_models(changed, compact_settings)
    assert original == rebuilt
    pd.testing.assert_frame_equal(table, changed_table)
