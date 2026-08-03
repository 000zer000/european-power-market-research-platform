"""Official price ingestion, caching, parsing and provenance."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from power_market.config import Settings

LOGGER = logging.getLogger(__name__)
USER_AGENT = "european-power-market-research-platform/1.0 (+research; contact via GitHub)"


class DataSourceError(RuntimeError):
    """Raised when an official payload cannot be safely interpreted."""


def sha256_bytes(content: bytes) -> str:
    """Return the lowercase SHA-256 digest for provenance."""
    return hashlib.sha256(content).hexdigest()


def _download(url: str, attempts: int = 4) -> bytes:
    """Download an HTTPS resource with bounded exponential retry."""
    error: Exception | None = None
    for attempt in range(attempts):
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=45) as response:
                if response.status != 200:
                    raise DataSourceError(f"unexpected HTTP status {response.status} for {url}")
                return bytes(response.read())
        except (HTTPError, URLError, TimeoutError, DataSourceError) as exc:
            error = exc
            if attempt + 1 < attempts:
                time.sleep(2**attempt)
    raise DataSourceError(f"failed to download {url}: {error}")


def parse_smard_payload(payload: bytes) -> pd.DataFrame:
    """Parse one SMARD chart payload without altering valid negative prices."""
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise DataSourceError("SMARD returned invalid JSON") from exc
    if not isinstance(data, dict) or set(data) != {"meta_data", "series"}:
        raise DataSourceError(f"SMARD schema changed: top-level keys={sorted(data)}")
    metadata = data["meta_data"]
    if not isinstance(metadata, dict) or not {"version", "created"} <= set(metadata):
        raise DataSourceError("SMARD meta_data schema changed")
    series = data["series"]
    if not isinstance(series, list):
        raise DataSourceError("SMARD series must be a list")
    rows: list[tuple[int, float]] = []
    for row in series:
        if not isinstance(row, list) or len(row) != 2:
            raise DataSourceError("SMARD series rows must be [unix_ms, value]")
        stamp, value = row
        if (
            not isinstance(stamp, int)
            or isinstance(value, bool)
            or (value is not None and not isinstance(value, int | float))
        ):
            raise DataSourceError("SMARD row has invalid timestamp or numeric value")
        rows.append((stamp, np.nan if value is None else float(value)))
    return pd.DataFrame(rows, columns=["unix_ms", "price_eur_per_mwh"])


def parse_energy_charts_payload(payload: bytes) -> pd.DataFrame:
    """Parse the documented Energy-Charts price response schema."""
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise DataSourceError("Energy-Charts returned invalid JSON") from exc
    if not isinstance(data, dict):
        raise DataSourceError("Energy-Charts response must be an object")
    required = {"unix_seconds", "price", "unit"}
    if not required <= set(data):
        raise DataSourceError(f"Energy-Charts schema missing {sorted(required - set(data))}")
    timestamps, prices = data["unix_seconds"], data["price"]
    if not isinstance(timestamps, list) or not isinstance(prices, list):
        raise DataSourceError("Energy-Charts timestamps and prices must be lists")
    if len(timestamps) != len(prices):
        raise DataSourceError("Energy-Charts timestamps and prices differ in length")
    normalised_unit = str(data["unit"]).replace(" ", "").upper()
    if normalised_unit not in {"EUR/MWH", "€/MWH"}:
        raise DataSourceError(f"unexpected Energy-Charts price unit: {data['unit']}")
    rows: list[tuple[int, float]] = []
    for stamp, value in zip(timestamps, prices, strict=True):
        if (
            not isinstance(stamp, int)
            or isinstance(value, bool)
            or (value is not None and not isinstance(value, int | float))
        ):
            raise DataSourceError("Energy-Charts contains an invalid observation")
        rows.append((stamp * 1000, np.nan if value is None else float(value)))
    return pd.DataFrame(rows, columns=["unix_ms", "price_eur_per_mwh"])


def _write_raw(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def _manifest_entry(url: str, path: Path, content: bytes, cached: bool) -> dict[str, Any]:
    downloaded_at = datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat()
    return {
        "source_url": url,
        "request_parameters": {},
        "downloaded_at_utc": downloaded_at,
        "manifested_at_utc": datetime.now(UTC).isoformat(),
        "cache_hit": cached,
        "local_cache_path": str(path.relative_to(path.parents[3])),
        "sha256": sha256_bytes(content),
        "unit": "EUR/MWh",
    }


def _load_or_download(url: str, path: Path, force: bool) -> tuple[bytes, bool]:
    if path.exists() and not force:
        return path.read_bytes(), True
    content = _download(url)
    _write_raw(path, content)
    return content, False


def _date_bounds_utc(settings: Settings) -> tuple[pd.Timestamp, pd.Timestamp]:
    timezone = ZoneInfo(settings.data.market_timezone)
    start = pd.Timestamp(datetime.combine(settings.data.start_date, datetime.min.time(), timezone))
    end_local = datetime.combine(
        settings.data.end_date + timedelta(days=1), datetime.min.time(), timezone
    )
    return start.tz_convert("UTC"), pd.Timestamp(end_local).tz_convert("UTC")


def fetch_smard(settings: Settings, force: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch official SMARD weekly hourly chunks covering the configured local dates."""
    raw_dir = settings.project_root / "data" / "raw" / "smard"
    series_id = settings.data.smard_series_id
    zone = settings.data.bidding_zone
    base = settings.data.smard_base_url.rstrip("/")
    index_url = f"{base}/{series_id}/{zone}/index_hour.json"
    index_path = raw_dir / "index_hour.json"
    index_content, index_cached = _load_or_download(index_url, index_path, force)
    try:
        index_data = json.loads(index_content)
        anchors = index_data["timestamps"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise DataSourceError("SMARD index schema changed") from exc
    if not isinstance(anchors, list) or not all(isinstance(item, int) for item in anchors):
        raise DataSourceError("SMARD index timestamps must be integer milliseconds")

    start_utc, end_utc = _date_bounds_utc(settings)
    lower_ms = int((start_utc - pd.Timedelta(days=8)).timestamp() * 1000)
    upper_ms = int(end_utc.timestamp() * 1000)
    selected = [stamp for stamp in anchors if lower_ms <= stamp < upper_ms]
    if not selected:
        raise DataSourceError("SMARD index contains no chunks for the requested period")

    entries = [_manifest_entry(index_url, index_path, index_content, index_cached)]
    frames: list[pd.DataFrame] = []
    for position, anchor in enumerate(selected, start=1):
        url = f"{base}/{series_id}/{zone}/{series_id}_{zone}_hour_{anchor}.json"
        path = raw_dir / f"{series_id}_{zone}_hour_{anchor}.json"
        content, cached = _load_or_download(url, path, force)
        entries.append(_manifest_entry(url, path, content, cached))
        frames.append(parse_smard_payload(content))
        if position % 25 == 0:
            LOGGER.info("downloaded_or_loaded_smard_chunks", extra={"completed": position})
    frame = pd.concat(frames, ignore_index=True)
    return _canonicalise(frame, settings), {
        "provider": "Bundesnetzagentur | SMARD.de",
        "series": "Day-ahead price, Germany/Luxembourg",
        "series_id": series_id,
        "licence": "CC BY 4.0",
        "licence_url": "https://www.smard.de/en/datennutzung",
        "entries": entries,
    }


def fetch_energy_charts(
    settings: Settings, force: bool = False
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch one documented Energy-Charts price response."""
    parameters = {
        "bzn": settings.data.bidding_zone,
        "start": settings.data.start_date.isoformat(),
        "end": settings.data.end_date.isoformat(),
    }
    url = f"{settings.data.energy_charts_url}?{urlencode(parameters)}"
    path = settings.project_root / "data" / "raw" / "energy_charts" / "prices.json"
    content, cached = _load_or_download(url, path, force)
    frame = parse_energy_charts_payload(content)
    entry = _manifest_entry(url, path, content, cached)
    entry["request_parameters"] = parameters
    return _canonicalise(frame, settings), {
        "provider": "Fraunhofer ISE Energy-Charts",
        "series": "Day-ahead price, Germany/Luxembourg",
        "licence": "See response licence_info and Energy-Charts terms",
        "entries": [entry],
    }


def synthetic_fixture(settings: Settings) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create deterministic non-market data for offline execution and tests."""
    start, end = _date_bounds_utc(settings)
    timestamps = pd.date_range(start, end, inclusive="left", freq="h")
    local = timestamps.tz_convert(settings.data.market_timezone)
    hour = local.hour.to_numpy()
    day = np.arange(len(timestamps), dtype=float) / 24.0
    price = (
        62.0
        + 22.0 * np.sin(2 * np.pi * (hour - 7) / 24)
        + 11.0 * np.sin(2 * np.pi * day / 7)
        + 8.0 * np.cos(2 * np.pi * day / 365.25)
        + 3.0 * np.sin(2 * np.pi * day / 31)
    )
    # Deterministic negative-price episodes and peaks exercise edge cases.
    price[(local.month == 5) & (hour >= 11) & (hour <= 14)] -= 75.0
    price[(local.month == 12) & (hour >= 17) & (hour <= 19)] += 55.0
    frame = pd.DataFrame({"timestamp_utc": timestamps, "price_eur_per_mwh": price})
    return frame, {
        "provider": "Deterministic synthetic fixture (not observed market data)",
        "formula_version": 1,
        "licence": "MIT project code; fixture contains no third-party observations",
        "entries": [],
    }


def _canonicalise(frame: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    frame = frame.copy()
    frame["timestamp_utc"] = pd.to_datetime(frame.pop("unix_ms"), unit="ms", utc=True)
    start, end = _date_bounds_utc(settings)
    return frame.loc[
        (frame["timestamp_utc"] >= start) & (frame["timestamp_utc"] < end)
    ].reset_index(drop=True)


def fetch_prices(settings: Settings, force: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch the configured source and persist canonical data plus provenance."""
    if settings.data.source == "smard":
        frame, manifest = fetch_smard(settings, force)
    elif settings.data.source == "energy_charts":
        frame, manifest = fetch_energy_charts(settings, force)
    else:
        frame, manifest = synthetic_fixture(settings)
    processed = settings.project_root / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    frame.to_csv(processed / "day_ahead_prices.csv", index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
    manifest.update(
        {
            "bidding_zone": settings.data.bidding_zone,
            "canonical_timezone": "UTC",
            "market_timezone": settings.data.market_timezone,
            "unit": settings.data.unit,
            "requested_start_local": settings.data.start_date.isoformat(),
            "requested_end_local_inclusive": settings.data.end_date.isoformat(),
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "observations": len(frame),
        }
    )
    destination = settings.project_root / "artifacts" / "provenance" / "manifest.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return frame, manifest


def load_prices(settings: Settings) -> pd.DataFrame:
    """Load canonical prices, preserving timezone awareness."""
    path = settings.project_root / "data" / "processed" / "day_ahead_prices.csv"
    frame = pd.read_csv(path)
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    return frame
