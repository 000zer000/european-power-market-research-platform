"""Chronological model selection and monthly expanding-window forecasts."""

from __future__ import annotations

import itertools
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# HistGradientBoosting uses OpenMP internally. Fixing its process-level worker
# count before importing scikit-learn makes executed research byte-reproducible.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

from power_market.config import Settings
from power_market.features import FEATURE_COLUMNS
from power_market.metrics import mae


@dataclass(frozen=True)
class SelectedModels:
    """Hyperparameters selected only on the validation period."""

    ridge_alpha: float
    tree_learning_rate: float
    tree_max_leaf_nodes: int


def _ridge(alpha: float) -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=alpha))])


def _tree(
    learning_rate: float, max_leaf_nodes: int, max_iter: int, seed: int
) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes,
        max_iter=max_iter,
        l2_regularization=1.0,
        random_state=seed,
        early_stopping=False,
    )


def _model_rows(features: pd.DataFrame, split: str | None = None) -> pd.DataFrame:
    mask = features[FEATURE_COLUMNS].notna().all(axis=1)
    if split is not None:
        mask &= features["split"].eq(split)
    return features.loc[mask]


def select_models(
    features: pd.DataFrame, settings: Settings
) -> tuple[SelectedModels, pd.DataFrame]:
    """Select ridge and tree settings on 2024 without reading 2025 outcomes."""
    train = _model_rows(features, "train")
    validation = _model_rows(features, "validation")
    x_train, y_train = train[FEATURE_COLUMNS], train["price_eur_per_mwh"]
    x_validation, y_validation = validation[FEATURE_COLUMNS], validation["price_eur_per_mwh"]
    rows: list[dict[str, Any]] = []
    with threadpool_limits(limits=1):
        for alpha in settings.forecast.ridge_alphas:
            model = _ridge(alpha).fit(x_train, y_train)
            rows.append(
                {
                    "model": "ridge",
                    "parameters": {"alpha": alpha},
                    "validation_mae": mae(y_validation, model.predict(x_validation)),
                }
            )
        for learning_rate, leaves in itertools.product(
            settings.forecast.tree_learning_rates, settings.forecast.tree_max_leaf_nodes
        ):
            model = _tree(
                learning_rate, leaves, settings.forecast.tree_max_iter, settings.random_seed
            ).fit(x_train, y_train)
            rows.append(
                {
                    "model": "hist_gradient_boosting",
                    "parameters": {
                        "learning_rate": learning_rate,
                        "max_leaf_nodes": leaves,
                    },
                    "validation_mae": mae(y_validation, model.predict(x_validation)),
                }
            )
    leaderboard = pd.DataFrame(rows).sort_values(["model", "validation_mae"]).reset_index(drop=True)
    ridge_row = leaderboard.loc[leaderboard["model"].eq("ridge")].iloc[0]
    tree_row = leaderboard.loc[leaderboard["model"].eq("hist_gradient_boosting")].iloc[0]
    selected = SelectedModels(
        ridge_alpha=float(ridge_row["parameters"]["alpha"]),
        tree_learning_rate=float(tree_row["parameters"]["learning_rate"]),
        tree_max_leaf_nodes=int(tree_row["parameters"]["max_leaf_nodes"]),
    )
    return selected, leaderboard


def walk_forward_predict(
    features: pd.DataFrame, selected: SelectedModels, settings: Settings
) -> pd.DataFrame:
    """Generate untouched-test predictions with monthly expanding retraining."""
    model_data = _model_rows(features)
    test = model_data.loc[model_data["split"].eq("test")].copy()
    output = test[
        [
            "timestamp_utc",
            "timestamp_local",
            "local_date",
            "local_hour",
            "month",
            "price_eur_per_mwh",
        ]
    ].rename(columns={"price_eur_per_mwh": "actual_price"})
    output["previous_day"] = test["lag_24h_price"].to_numpy()
    output["previous_week"] = test["lag_168h_price"].to_numpy()
    output["ridge"] = np.nan
    output["hist_gradient_boosting"] = np.nan

    month_periods = test["timestamp_utc"].dt.tz_localize(None).dt.to_period("M")
    with threadpool_limits(limits=1):
        for period in month_periods.unique():
            predict_mask = month_periods.eq(period)
            month_rows = test.loc[predict_mask]
            origin = month_rows["timestamp_utc"].min()
            history = model_data.loc[model_data["timestamp_utc"] < origin]
            ridge = _ridge(selected.ridge_alpha).fit(
                history[FEATURE_COLUMNS], history["price_eur_per_mwh"]
            )
            tree = _tree(
                selected.tree_learning_rate,
                selected.tree_max_leaf_nodes,
                settings.forecast.tree_max_iter,
                settings.random_seed,
            ).fit(history[FEATURE_COLUMNS], history["price_eur_per_mwh"])
            output.loc[predict_mask, "ridge"] = ridge.predict(month_rows[FEATURE_COLUMNS])
            output.loc[predict_mask, "hist_gradient_boosting"] = tree.predict(
                month_rows[FEATURE_COLUMNS]
            )
    if output.isna().any().any():
        raise ValueError("walk-forward prediction produced missing values")
    return output.reset_index(drop=True)


def train_and_predict(
    features: pd.DataFrame, settings: Settings
) -> tuple[pd.DataFrame, SelectedModels]:
    """Run validation-only selection and persist final test predictions."""
    selected, leaderboard = select_models(features, settings)
    predictions = walk_forward_predict(features, selected, settings)
    metrics_dir = settings.project_root / "artifacts" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    leaderboard.assign(parameters=leaderboard["parameters"].map(json.dumps)).to_csv(
        metrics_dir / "validation_model_selection.csv", index=False
    )
    selection_payload = {
        "selection_period": "2024-01-01 through 2024-12-31",
        "final_test_period_not_used": "2025-01-01 through 2025-12-31",
        "retraining": "Monthly expanding window during final test evaluation",
        "ridge_alpha": selected.ridge_alpha,
        "tree_learning_rate": selected.tree_learning_rate,
        "tree_max_leaf_nodes": selected.tree_max_leaf_nodes,
        "random_seed": settings.random_seed,
    }
    (metrics_dir / "selected_models.json").write_text(
        json.dumps(selection_payload, indent=2), encoding="utf-8"
    )
    predictions.to_csv(metrics_dir / "test_predictions.csv", index=False)
    return predictions, selected


def load_predictions(project_root: Path) -> pd.DataFrame:
    """Load saved final-test predictions."""
    frame = pd.read_csv(project_root / "artifacts" / "metrics" / "test_predictions.csv")
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    return frame
