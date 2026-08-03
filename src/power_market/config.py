"""Validated configuration loading."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Base class that rejects misspelled or unknown settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class DataConfig(StrictModel):
    """Data-source and market-time settings."""

    source: Literal["smard", "energy_charts", "fixture"]
    bidding_zone: str = "DE-LU"
    start_date: date
    end_date: date
    market_timezone: str = "Europe/Berlin"
    unit: Literal["EUR/MWh"] = "EUR/MWh"
    smard_base_url: str
    smard_series_id: int = 4169
    energy_charts_url: str

    @model_validator(mode="after")
    def dates_are_ordered(self) -> DataConfig:
        """Reject an inverted extraction range."""
        if self.start_date > self.end_date:
            raise ValueError("data.start_date must not be after data.end_date")
        return self


class SplitConfig(StrictModel):
    """Chronological experiment boundaries, inclusive by local delivery date."""

    train_end: date
    validation_end: date
    test_end: date

    @model_validator(mode="after")
    def dates_are_ordered(self) -> SplitConfig:
        """Require strict temporal separation."""
        if not self.train_end < self.validation_end < self.test_end:
            raise ValueError("split dates must satisfy train < validation < test")
        return self


class ForecastConfig(StrictModel):
    """Deterministic model-selection settings."""

    ridge_alphas: list[float] = Field(min_length=1)
    tree_learning_rates: list[float] = Field(min_length=1)
    tree_max_leaf_nodes: list[int] = Field(min_length=1)
    tree_max_iter: int = Field(ge=10)
    retraining_frequency: Literal["monthly"] = "monthly"
    bootstrap_samples: int = Field(ge=10)


class BatteryConfig(StrictModel):
    """Battery power, energy, efficiency and cost assumptions."""

    energy_capacity_mwh: float = Field(gt=0)
    max_charge_power_mw: float = Field(gt=0)
    max_discharge_power_mw: float = Field(gt=0)
    min_soc_mwh: float = Field(ge=0)
    max_soc_mwh: float = Field(gt=0)
    initial_soc_mwh: float = Field(ge=0)
    terminal_soc_mwh: float = Field(ge=0)
    charge_efficiency: float = Field(gt=0, le=1)
    discharge_efficiency: float = Field(gt=0, le=1)
    degradation_cost_eur_per_mwh: float = Field(ge=0)

    @model_validator(mode="after")
    def physical_bounds_are_consistent(self) -> BatteryConfig:
        """Reject energy bounds or endpoints outside physical capacity."""
        if self.max_soc_mwh > self.energy_capacity_mwh:
            raise ValueError("max_soc_mwh cannot exceed energy_capacity_mwh")
        if self.min_soc_mwh >= self.max_soc_mwh:
            raise ValueError("min_soc_mwh must be smaller than max_soc_mwh")
        if not self.min_soc_mwh <= self.initial_soc_mwh <= self.max_soc_mwh:
            raise ValueError("initial_soc_mwh lies outside state-of-charge bounds")
        if not self.min_soc_mwh <= self.terminal_soc_mwh <= self.max_soc_mwh:
            raise ValueError("terminal_soc_mwh lies outside state-of-charge bounds")
        return self


class Settings(StrictModel):
    """Complete experiment configuration."""

    random_seed: int
    data: DataConfig
    splits: SplitConfig
    forecast: ForecastConfig
    battery: BatteryConfig
    project_root: Path = Field(exclude=True)


def load_settings(path: str | Path) -> Settings:
    """Load YAML and validate all fields, resolving the repository root."""
    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("configuration root must be a mapping")
    payload["project_root"] = config_path.parent.parent
    return Settings.model_validate(payload)
