from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import power_market.data as data_module
from power_market.data import (
    DataSourceError,
    fetch_energy_charts,
    fetch_smard,
    load_prices,
    parse_energy_charts_payload,
    parse_smard_payload,
    sha256_bytes,
    synthetic_fixture,
)


def test_parse_official_smard_shape_and_negative_price() -> None:
    payload = (Path(__file__).parent / "fixtures" / "smard_hour_sample.json").read_bytes()
    frame = parse_smard_payload(payload)
    assert frame["unix_ms"].tolist() == [1672531200000, 1672534800000, 1672538400000]
    assert frame["price_eur_per_mwh"].tolist() == [12.5, 0.0, -3.25]


@pytest.mark.parametrize(
    "payload,match",
    [
        (b"not-json", "invalid JSON"),
        (b'{"series": []}', "schema changed"),
        (b'{"meta_data":{"version":1,"created":1},"series":[[1]]}', "rows"),
    ],
)
def test_smard_schema_changes_fail_loudly(payload: bytes, match: str) -> None:
    with pytest.raises(DataSourceError, match=match):
        parse_smard_payload(payload)


def test_energy_charts_parser_checks_unit_and_lengths() -> None:
    payload = json.dumps(
        {
            "unix_seconds": [1, 2],
            "price": [-1.0, 4.0],
            "unit": "EUR / MWh",
            "license_info": "x",
            "deprecated": False,
        }
    ).encode()
    frame = parse_energy_charts_payload(payload)
    assert frame.iloc[0].to_dict() == {"unix_ms": 1000.0, "price_eur_per_mwh": -1.0}
    with pytest.raises(DataSourceError, match="differ"):
        parse_energy_charts_payload(b'{"unix_seconds":[1],"price":[],"unit":"EUR/MWh"}')
    with pytest.raises(DataSourceError, match="unit"):
        parse_energy_charts_payload(b'{"unix_seconds":[1],"price":[2],"unit":"MW"}')


def test_nullable_official_value_is_left_for_validation() -> None:
    payload = b'{"unix_seconds":[1],"price":[null],"unit":"EUR/MWh"}'
    frame = parse_energy_charts_payload(payload)
    assert np.isnan(frame["price_eur_per_mwh"].iloc[0])


def test_hash_is_stable() -> None:
    assert (
        sha256_bytes(b"abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_synthetic_fixture_is_deterministic_and_negative(settings) -> None:
    first, manifest = synthetic_fixture(settings)
    second, _ = synthetic_fixture(settings)
    assert first.equals(second)
    assert np.isfinite(first["price_eur_per_mwh"]).all()
    assert (first["price_eur_per_mwh"] < 0).any()
    assert "synthetic" in manifest["provider"].lower()


def test_smard_fetches_weekly_chunk_then_uses_cache(settings, monkeypatch) -> None:
    day = pd.Timestamp("2023-01-01").date()
    scoped_data = settings.data.model_copy(
        update={"source": "smard", "start_date": day, "end_date": day}
    )
    scoped = settings.model_copy(update={"data": scoped_data})
    anchor = pd.Timestamp("2022-12-25T23:00:00Z")
    timestamps = pd.date_range(anchor, periods=7 * 24, freq="h")
    index_payload = json.dumps({"timestamps": [int(anchor.timestamp() * 1000)]}).encode()
    chunk_payload = json.dumps(
        {
            "meta_data": {"version": 1, "created": 1},
            "series": [
                [int(stamp.timestamp() * 1000), float(position - 10)]
                for position, stamp in enumerate(timestamps)
            ],
        }
    ).encode()
    calls: list[str] = []

    def fake_download(url: str, attempts: int = 4) -> bytes:
        calls.append(url)
        return index_payload if url.endswith("index_hour.json") else chunk_payload

    monkeypatch.setattr(data_module, "_download", fake_download)
    frame, manifest = fetch_smard(scoped)
    assert len(frame) == 24
    assert frame["timestamp_utc"].is_unique
    assert len(calls) == 2
    assert all(not entry["cache_hit"] for entry in manifest["entries"])
    cached, cached_manifest = fetch_smard(scoped)
    assert cached.equals(frame)
    assert len(calls) == 2
    assert all(entry["cache_hit"] for entry in cached_manifest["entries"])


def test_energy_charts_fetch_persists_and_loads(settings, monkeypatch) -> None:
    day = pd.Timestamp("2023-01-01").date()
    scoped_data = settings.data.model_copy(
        update={"source": "energy_charts", "start_date": day, "end_date": day}
    )
    scoped = settings.model_copy(update={"data": scoped_data})
    timestamps = pd.date_range("2022-12-31T23:00:00Z", periods=24, freq="h")
    payload = json.dumps(
        {
            "unix_seconds": [int(stamp.timestamp()) for stamp in timestamps],
            "price": [float(value) for value in range(24)],
            "unit": "EUR / MWh",
            "license_info": "CC BY 4.0",
        }
    ).encode()
    monkeypatch.setattr(data_module, "_download", lambda url: payload)
    frame, manifest = fetch_energy_charts(scoped)
    assert len(frame) == 24
    assert manifest["entries"][0]["request_parameters"]["bzn"] == "DE-LU"
    data_module.fetch_prices(scoped)
    assert load_prices(scoped).equals(frame)
