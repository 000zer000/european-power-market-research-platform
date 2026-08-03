"""Command-line interface for every reproducible research stage."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

from power_market.config import load_settings
from power_market.pipeline import (
    evaluate_stage,
    feature_stage,
    fetch_stage,
    optimise_stage,
    report_stage,
    run_all,
    train_stage,
    validate_stage,
)


class JsonFormatter(logging.Formatter):
    """Compact structured logs suitable for local runs and CI."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        return json.dumps(payload)


def _configure_logging(verbose: bool) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(logging.DEBUG if verbose else logging.INFO)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/default.yaml", help="validated YAML configuration"
    )
    parser.add_argument("--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)
    fetch = subparsers.add_parser("fetch-data")
    fetch.add_argument("--force", action="store_true", help="replace cached raw responses")
    for command in ["validate-data", "build-features", "train", "evaluate", "optimise", "report"]:
        subparsers.add_parser(command)
    run = subparsers.add_parser("run-all")
    run.add_argument("--force", action="store_true", help="replace cached raw responses")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Execute one CLI stage and return a process status."""
    arguments = _parser().parse_args(argv)
    _configure_logging(arguments.verbose)
    settings = load_settings(arguments.config)
    actions: dict[str, Any] = {
        "fetch-data": lambda: {"rows": len(fetch_stage(settings, arguments.force))},
        "validate-data": lambda: validate_stage(settings),
        "build-features": lambda: {"rows": len(feature_stage(settings))},
        "train": lambda: {"test_predictions": len(train_stage(settings))},
        "evaluate": lambda: evaluate_stage(settings),
        "optimise": lambda: {"strategies": len(optimise_stage(settings)[1])},
        "report": lambda: {"outputs": report_stage(settings)},
        "run-all": lambda: run_all(settings, arguments.force),
    }
    result = actions[arguments.command]()
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
