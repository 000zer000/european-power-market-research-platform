"""Adversarial schedules exercise the auditor independently of the solver."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from power_market.battery import (
    OptimisationError,
    audit_schedule,
    evaluate_schedule,
    optimise_schedule,
)


@pytest.fixture
def battery(settings):
    return settings.battery.model_copy(
        update={
            "energy_capacity_mwh": 1.0,
            "max_charge_power_mw": 1.0,
            "max_discharge_power_mw": 1.0,
            "min_soc_mwh": 0.0,
            "max_soc_mwh": 1.0,
            "initial_soc_mwh": 0.0,
            "terminal_soc_mwh": 0.0,
            "charge_efficiency": 1.0,
            "discharge_efficiency": 1.0,
            "degradation_cost_eur_per_mwh": 2.0,
        }
    )


def test_discontinuous_energy_is_rejected(battery):
    schedule = pd.DataFrame(
        {
            "timestamp": [0, 1, 2],
            "duration_hours": [1, 1, 1],
            "charge_mw": [0, 0, 0],
            "discharge_mw": [0, 1, 0],
            "soc_start_mwh": [0, 1, 0],
            "soc_end_mwh": [0, 0, 0],
            "internal_throughput_mwh": [0, 1, 0],
        }
    )
    with pytest.raises(OptimisationError, match="continuity"):
        evaluate_schedule(schedule, [100, 100, 100], battery)


def test_forged_throughput_is_rejected(battery):
    schedule = optimise_schedule([0, 100], battery).schedule
    schedule["internal_throughput_mwh"] = 0.0
    with pytest.raises(OptimisationError, match="throughput"):
        evaluate_schedule(schedule, [0, 100], battery)


def test_audit_and_settlement_use_the_same_numeric_conversion(battery):
    schedule = optimise_schedule([0, 100], battery).schedule
    expected = evaluate_schedule(schedule, [0, 100], battery)
    for column in schedule.columns.difference(["timestamp"]):
        schedule[column] = schedule[column].astype(str)
    assert evaluate_schedule(schedule, [0, 100], battery) == expected


@pytest.mark.parametrize(
    "column",
    [
        "charge_mw",
        "discharge_mw",
        "duration_hours",
        "soc_start_mwh",
        "soc_end_mwh",
        "internal_throughput_mwh",
        "decision_price_eur_per_mwh",
        "charge_mode",
    ],
)
@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_nonfinite_schedule_columns_rejected(battery, column, invalid):
    schedule = optimise_schedule([0, 100], battery).schedule
    schedule[column] = schedule[column].astype(float)
    schedule.loc[0, column] = invalid
    with pytest.raises(OptimisationError, match="finite"):
        audit_schedule(schedule, battery)


@pytest.mark.parametrize("duration", [0, -1])
def test_nonpositive_duration_rejected(battery, duration):
    schedule = optimise_schedule([1, 1], battery).schedule
    schedule.loc[0, "duration_hours"] = duration
    with pytest.raises(OptimisationError, match="positive"):
        audit_schedule(schedule, battery)


def test_missing_and_empty_schedule_rejected(battery):
    schedule = optimise_schedule([0, 100], battery).schedule
    with pytest.raises(OptimisationError, match="nonempty"):
        audit_schedule(schedule.iloc[:0], battery)
    with pytest.raises(OptimisationError, match="missing"):
        audit_schedule(schedule.drop(columns="soc_start_mwh"), battery)


def test_negative_prices_asymmetric_efficiencies_and_half_hours(battery):
    battery = battery.model_copy(update={"charge_efficiency": 0.8, "discharge_efficiency": 0.9})
    stamps = pd.date_range("2025-01-01", periods=2, freq="30min", tz="UTC")
    schedule = optimise_schedule([-50, 100], battery, [0.5, 0.5], stamps).schedule
    value = evaluate_schedule(schedule, pd.Series([-50, 100], index=stamps), battery)
    assert value["internal_throughput_mwh"] == pytest.approx(0.8)
    assert value["degradation_cost_eur"] == pytest.approx(1.6)
    assert value["realised_gross_market_revenue_eur"] == pytest.approx(61.0)
    assert value["equivalent_full_cycles"] == pytest.approx(0.4)


def test_actual_price_timestamp_alignment_rejected(battery):
    stamps = pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
    schedule = optimise_schedule([0, 100], battery, timestamps=stamps).schedule
    with pytest.raises(ValueError, match="timestamp"):
        evaluate_schedule(schedule, pd.Series([0, 100], index=stamps[::-1]), battery)


@pytest.mark.parametrize(
    "stamps",
    [
        ["2025-01-01T00:00Z", "2025-01-01T00:00Z"],
        ["2025-01-01T01:00Z", "2025-01-01T00:00Z"],
        ["2025-01-01T00:00Z", "2025-01-01T02:00Z"],
        ["2025-01-01T00:00Z", None],
    ],
)
def test_invalid_timestamps_rejected(battery, stamps):
    schedule = optimise_schedule([0, 100], battery).schedule
    schedule["timestamp"] = stamps
    with pytest.raises(OptimisationError, match="timestamp"):
        audit_schedule(schedule, battery)


@pytest.mark.parametrize("date", ["2025-03-30", "2025-10-26"])
def test_real_dst_day_is_accepted(battery, date):
    start = pd.Timestamp(date, tz="Europe/Berlin")
    end = start + pd.DateOffset(days=1)
    stamps = pd.date_range(start, end, freq="h", inclusive="left")
    schedule = optimise_schedule(
        np.sin(np.arange(len(stamps))), battery, timestamps=stamps
    ).schedule
    audit_schedule(schedule, battery)
    assert len(schedule) in (23, 25)


@pytest.mark.parametrize("column,value", [("soc_start_mwh", 2.0), ("soc_end_mwh", -1.0)])
def test_both_sides_of_all_soc_bounds_are_checked(battery, column, value):
    schedule = optimise_schedule([0, 100], battery).schedule
    schedule.loc[1 if column == "soc_start_mwh" else 0, column] = value
    with pytest.raises(OptimisationError, match="state-of-charge bounds"):
        audit_schedule(schedule, battery)


@pytest.mark.parametrize("tolerance", [0, -1, np.nan, np.inf])
def test_invalid_tolerance_rejected(battery, tolerance):
    schedule = optimise_schedule([0, 100], battery).schedule
    with pytest.raises(ValueError, match="tolerance"):
        audit_schedule(schedule, battery, tolerance)


def test_relative_tolerance_does_not_hide_large_state_errors(battery):
    battery = battery.model_copy(
        update={
            "energy_capacity_mwh": 100000.0,
            "max_soc_mwh": 100000.0,
            "initial_soc_mwh": 50000.0,
            "terminal_soc_mwh": 50000.0,
        }
    )
    schedule = optimise_schedule([1, 1], battery).schedule
    schedule.loc[0, "soc_end_mwh"] += 0.1
    with pytest.raises(OptimisationError):
        audit_schedule(schedule, battery)
