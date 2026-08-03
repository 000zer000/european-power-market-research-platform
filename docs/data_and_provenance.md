# Data and provenance

## Executed source

The executed analysis uses the public hourly day-ahead price series for the Germany/Luxembourg bidding zone from [Bundesnetzagentur SMARD](https://www.smard.de/en/downloadcenter), series `4169`, for local delivery dates 2023-01-01 through 2025-12-31. SMARD defines the series as the weighted wholesale electricity price resulting from the previous day's auction and reports it in EUR/MWh. Its data originate from ENTSO-E and are checked by Bundesnetzagentur. The official [SMARD data-use page](https://www.smard.de/en/datennutzung) licenses downloadable market data under CC BY 4.0 and requires attribution to “Bundesnetzagentur | SMARD.de”.

Fraunhofer ISE [Energy-Charts API](https://api.energy-charts.info/) was the preferred adapter. Its documented `/price` endpoint was returning HTTP 503 on 2026-08-03 for the API root, OpenAPI document and representative DE-LU requests after retry. The code retains a validated Energy-Charts adapter, but the executed evidence uses SMARD because it was the available authoritative European source. This decision and every exact URL are recorded in the manifest.

## Resolution adjustment after 1 October 2025

The European Single Day-Ahead Coupling market time unit changed from hourly to 15 minutes for delivery beginning 2025-10-01, as documented by [ENTSO-E](https://www.entsoe.eu/network_codes/cacm/implementation/sdac/) and [EPEX SPOT](https://www.epexspot.com/en/new-15-minute-products-market-coupling). SMARD continues to publish an official `hour` series; the platform uses that series to keep one uniform hourly decision interval through 2025. For a one-hour battery schedule held constant across the hour, the hour price represents the four equal-duration quarter-hour products. This sacrifices intrahour dispatch detail and is stated as a limitation.

## Timestamp contract

- The canonical key is a timezone-aware UTC timestamp.
- The market timezone is `Europe/Berlin` and is used only to derive delivery date, hour, weekday, month, weekend and holiday fields.
- Spring DST days contain 23 elapsed hourly intervals.
- Autumn DST days contain 25 elapsed hourly intervals. The two local 02:00 observations have different UTC offsets and distinct UTC keys.
- No naïve local timestamp is used as a join key.

For date-only Energy-Charts requests, the official API specifies local 00:00 for `start` and local 23:59 for `end`; returned Unix seconds are UTC. SMARD chart payload timestamps are Unix milliseconds and are likewise canonicalised to UTC.

## Cache and manifest

Each network response is written byte-for-byte to `data/raw/<provider>/`. Unless `--force` is supplied, reruns read these immutable historical responses. `artifacts/provenance/manifest.json` records:

- provider and series identity;
- source URL and request parameters;
- retrieval time in UTC;
- cache path and whether it was a cache hit;
- SHA-256 of the exact response;
- requested dates, bidding zone, timezone and unit;
- applicable licence and attribution link.

The manifest deliberately does not claim that a cached file was freshly downloaded. Raw market files are ignored by Git to avoid repository bloat; the deterministic synthetic fixture and derived evidence are sufficient for offline tests and review.

## Validation and non-repair policy

`validation.py` checks required columns, parseable timezone-aware timestamps, uniqueness, monotonic order, finite numeric values, exact UTC-hour coverage, no out-of-range rows, and valid 23/24/25-hour local days. The report includes counts and examples. It explicitly does not interpolate, clip, impute, deduplicate, drop or modify observations. Negative prices are legitimate market observations.

## Licence boundary

Project code and original documentation use the MIT licence. Source market data and data-derived artifacts retain their CC BY 4.0 attribution. The repository does not include proprietary exchange downloads, credentials, employer data or confidential information.
