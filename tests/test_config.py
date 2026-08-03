from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from power_market.config import load_settings


def test_default_configuration_is_valid() -> None:
    settings = load_settings(Path(__file__).parents[1] / "configs" / "default.yaml")
    assert settings.data.bidding_zone == "DE-LU"
    assert settings.battery.charge_efficiency == 0.95


def test_unknown_setting_is_rejected(tmp_path: Path) -> None:
    source = Path(__file__).parents[1] / "configs" / "offline.yaml"
    payload = yaml.safe_load(source.read_text())
    payload["unexpected"] = True
    config = tmp_path / "bad.yaml"
    config.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValidationError, match="unexpected"):
        load_settings(config)


def test_inconsistent_battery_bounds_are_rejected(tmp_path: Path) -> None:
    source = Path(__file__).parents[1] / "configs" / "offline.yaml"
    payload = yaml.safe_load(source.read_text())
    payload["battery"]["max_soc_mwh"] = 3.0
    config = tmp_path / "bad.yaml"
    config.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValidationError, match="capacity"):
        load_settings(config)
