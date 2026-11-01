"""Mixed-integer battery scheduling and independent realised-value evaluation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from power_market.config import BatteryConfig


class OptimisationError(RuntimeError):
    """Raised when the battery problem is invalid, infeasible or not solved optimally."""


@dataclass(frozen=True)
class ScheduleResult:
    """Optimal physical schedule and solver evidence."""

    schedule: pd.DataFrame
    forecast_net_value_eur: float
    solver_status: int
    solver_message: str
    mip_gap: float


def _finite_vector(values: Iterable[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a nonempty finite one-dimensional vector")
    return array


def optimise_schedule(
    decision_prices: Iterable[float],
    battery: BatteryConfig,
    durations_hours: Iterable[float] | None = None,
    timestamps: Iterable[object] | None = None,
) -> ScheduleResult:
    """Optimise one horizon while preventing simultaneous charge and discharge."""
    prices = _finite_vector(decision_prices, "decision_prices")
    periods = len(prices)
    durations = (
        np.ones(periods, dtype=float)
        if durations_hours is None
        else _finite_vector(durations_hours, "durations_hours")
    )
    if durations.shape != prices.shape or (durations <= 0).any():
        raise ValueError("durations_hours must be positive and match prices")
    if timestamps is None:
        stamp_array = np.arange(periods)
    else:
        stamp_array = np.asarray(list(timestamps), dtype=object)
        if stamp_array.shape != prices.shape:
            raise ValueError("timestamps must match prices")

    charge_start = 0
    discharge_start = periods
    soc_start = 2 * periods
    mode_start = 3 * periods + 1
    variable_count = 4 * periods + 1

    objective = np.zeros(variable_count)
    degradation = battery.degradation_cost_eur_per_mwh
    objective[charge_start:discharge_start] = durations * (
        prices + degradation * battery.charge_efficiency
    )
    objective[discharge_start:soc_start] = durations * (
        -prices + degradation / battery.discharge_efficiency
    )

    lower = np.zeros(variable_count)
    upper = np.empty(variable_count)
    upper[charge_start:discharge_start] = battery.max_charge_power_mw
    upper[discharge_start:soc_start] = battery.max_discharge_power_mw
    lower[soc_start:mode_start] = battery.min_soc_mwh
    upper[soc_start:mode_start] = battery.max_soc_mwh
    lower[soc_start] = upper[soc_start] = battery.initial_soc_mwh
    lower[mode_start - 1] = upper[mode_start - 1] = battery.terminal_soc_mwh
    upper[mode_start:] = 1.0

    # T dynamics equalities plus two tight mode inequalities per interval.
    matrix = lil_matrix((3 * periods, variable_count), dtype=float)
    constraint_lower = np.full(3 * periods, -np.inf)
    constraint_upper = np.full(3 * periods, np.inf)
    for period in range(periods):
        duration = durations[period]
        matrix[period, charge_start + period] = -battery.charge_efficiency * duration
        matrix[period, discharge_start + period] = duration / battery.discharge_efficiency
        matrix[period, soc_start + period] = -1.0
        matrix[period, soc_start + period + 1] = 1.0
        constraint_lower[period] = constraint_upper[period] = 0.0

        charge_row = periods + period
        matrix[charge_row, charge_start + period] = 1.0
        matrix[charge_row, mode_start + period] = -battery.max_charge_power_mw
        constraint_upper[charge_row] = 0.0

        discharge_row = 2 * periods + period
        matrix[discharge_row, discharge_start + period] = 1.0
        matrix[discharge_row, mode_start + period] = battery.max_discharge_power_mw
        constraint_upper[discharge_row] = battery.max_discharge_power_mw

    integrality = np.zeros(variable_count, dtype=int)
    integrality[mode_start:] = 1
    result = milp(
        objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(matrix.tocsr(), constraint_lower, constraint_upper),
        options={"presolve": True, "mip_rel_gap": 0.0},
    )
    if not result.success or result.status != 0 or result.x is None:
        raise OptimisationError(f"battery MILP did not reach an optimum: {result.message}")
    charge = np.maximum(result.x[charge_start:discharge_start], 0.0)
    discharge = np.maximum(result.x[discharge_start:soc_start], 0.0)
    soc = result.x[soc_start:mode_start]
    mode = np.rint(result.x[mode_start:]).astype(int)
    throughput = durations * (
        battery.charge_efficiency * charge + discharge / battery.discharge_efficiency
    )
    schedule = pd.DataFrame(
        {
            "timestamp": stamp_array,
            "duration_hours": durations,
            "decision_price_eur_per_mwh": prices,
            "charge_mw": charge,
            "discharge_mw": discharge,
            "soc_start_mwh": soc[:-1],
            "soc_end_mwh": soc[1:],
            "charge_mode": mode,
            "internal_throughput_mwh": throughput,
        }
    )
    audit_schedule(schedule, battery)
    reconstructed = float(
        np.sum(prices * (discharge - charge) * durations - degradation * throughput)
    )
    if not np.isclose(reconstructed, -float(result.fun), atol=1e-5):
        raise OptimisationError("solver objective failed independent reconstruction")
    mip_gap = float(getattr(result, "mip_gap", 0.0) or 0.0)
    return ScheduleResult(schedule, reconstructed, result.status, result.message, mip_gap)


def audit_schedule(schedule: pd.DataFrame, battery: BatteryConfig, tolerance: float = 1e-6) -> None:
    """Verify an external schedule using absolute MW/MWh tolerances, never relative ones.

    Datetime starts must be contiguous in UTC with the declared durations. Numeric
    timestamps are strictly increasing interval identifiers, not elapsed hours.
    Derived throughput is checked against physical flows, not trusted as input.
    """
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    required = {
        "timestamp",
        "duration_hours",
        "charge_mw",
        "discharge_mw",
        "soc_start_mwh",
        "soc_end_mwh",
        "internal_throughput_mwh",
    }
    if schedule.empty:
        raise OptimisationError("schedule must be nonempty")
    if not schedule.columns.is_unique:
        raise OptimisationError("schedule columns must be unique")
    missing = required.difference(schedule.columns)
    if missing:
        raise OptimisationError(f"schedule is missing columns: {sorted(missing)}")
    numeric = required - {"timestamp"}
    numeric.update({"decision_price_eur_per_mwh", "charge_mode"}.intersection(schedule.columns))
    for column in sorted(numeric):
        try:
            values = schedule[column].to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise OptimisationError(f"{column} must contain finite numeric values") from exc
        if not np.isfinite(values).all():
            raise OptimisationError(f"{column} must contain finite numeric values")
    charge = schedule["charge_mw"].to_numpy(dtype=float)
    discharge = schedule["discharge_mw"].to_numpy(dtype=float)
    duration = schedule["duration_hours"].to_numpy(dtype=float)
    start = schedule["soc_start_mwh"].to_numpy(dtype=float)
    end = schedule["soc_end_mwh"].to_numpy(dtype=float)
    if (duration <= 0).any():
        raise OptimisationError("duration_hours must be positive")
    _audit_timestamps(schedule, duration)
    if (
        (start < battery.min_soc_mwh - tolerance).any()
        or (start > battery.max_soc_mwh + tolerance).any()
        or (end < battery.min_soc_mwh - tolerance).any()
        or (end > battery.max_soc_mwh + tolerance).any()
    ):
        raise OptimisationError("schedule violates state-of-charge bounds")
    if not np.allclose(end[:-1], start[1:], atol=tolerance, rtol=0):
        raise OptimisationError("schedule violates state-of-charge continuity")
    expected_end = (
        start
        + battery.charge_efficiency * charge * duration
        - discharge * duration / battery.discharge_efficiency
    )
    checks = [
        np.allclose(end, expected_end, atol=tolerance, rtol=0),
        (charge >= -tolerance).all() and (charge <= battery.max_charge_power_mw + tolerance).all(),
        (discharge >= -tolerance).all()
        and (discharge <= battery.max_discharge_power_mw + tolerance).all(),
        not ((charge > tolerance) & (discharge > tolerance)).any(),
        abs(start[0] - battery.initial_soc_mwh) <= tolerance,
        abs(end[-1] - battery.terminal_soc_mwh) <= tolerance,
    ]
    if not all(checks):
        raise OptimisationError(f"post-solve physical audit failed: {checks}")
    throughput = duration * (
        battery.charge_efficiency * charge + discharge / battery.discharge_efficiency
    )
    if not np.isfinite(throughput).all() or not np.allclose(
        schedule["internal_throughput_mwh"].to_numpy(dtype=float),
        throughput,
        atol=tolerance,
        rtol=0,
    ):
        raise OptimisationError("stored internal throughput disagrees with physical flows")
    if "charge_mode" in schedule:
        mode = schedule["charge_mode"].to_numpy(dtype=float)
        if not (
            (
                np.isclose(mode, 0, atol=tolerance, rtol=0)
                | np.isclose(mode, 1, atol=tolerance, rtol=0)
            ).all()
            and (charge <= battery.max_charge_power_mw * mode + tolerance).all()
            and (discharge <= battery.max_discharge_power_mw * (1 - mode) + tolerance).all()
        ):
            raise OptimisationError("charge_mode must be binary and consistent with power flows")


def _audit_timestamps(schedule: pd.DataFrame, durations: np.ndarray) -> None:
    stamps = schedule["timestamp"]
    if pd.api.types.is_numeric_dtype(stamps.dtype):
        values = stamps.to_numpy(dtype=float)
        if not np.isfinite(values).all() or not (np.diff(values) > 0).all():
            raise OptimisationError("numeric timestamps must be finite and strictly increasing")
        return
    try:
        utc = pd.DatetimeIndex(pd.to_datetime(stamps, utc=True, errors="raise"))
    except (TypeError, ValueError) as exc:
        raise OptimisationError("timestamps must be valid datetime interval starts") from exc
    if utc.hasnans or utc.has_duplicates or not utc.is_monotonic_increasing:
        raise OptimisationError("timestamps must be present, unique and strictly increasing")
    elapsed_hours = np.asarray((utc[1:] - utc[:-1]).total_seconds(), dtype=float) / 3600
    if not np.allclose(elapsed_hours, durations[:-1], atol=1e-9, rtol=0):
        raise OptimisationError("timestamp spacing must match duration_hours in UTC")


def evaluate_schedule(
    schedule: pd.DataFrame, actual_prices: Iterable[float], battery: BatteryConfig
) -> dict[str, float]:
    """Value a fixed forecast-chosen schedule against realised prices."""
    actual = _finite_vector(actual_prices, "actual_prices")
    if len(actual) != len(schedule):
        raise ValueError("actual_prices must match every scheduled interval")
    audit_schedule(schedule, battery)
    if isinstance(actual_prices, pd.Series) and isinstance(actual_prices.index, pd.DatetimeIndex):
        stamps = pd.DatetimeIndex(pd.to_datetime(schedule["timestamp"], utc=True))
        if not stamps.equals(actual_prices.index.tz_convert("UTC")):
            raise ValueError("actual_prices timestamps must match the schedule in order")
    duration = schedule["duration_hours"].to_numpy()
    gross = float(
        np.sum(
            actual
            * (schedule["discharge_mw"].to_numpy() - schedule["charge_mw"].to_numpy())
            * duration
        )
    )
    throughput = float(
        np.sum(
            duration
            * (
                battery.charge_efficiency * schedule["charge_mw"].to_numpy(dtype=float)
                + schedule["discharge_mw"].to_numpy(dtype=float) / battery.discharge_efficiency
            )
        )
    )
    degradation = battery.degradation_cost_eur_per_mwh * throughput
    if not np.isfinite([gross, throughput, degradation, gross - degradation]).all():
        raise OptimisationError("realised schedule accounting must be finite")
    usable_range = battery.max_soc_mwh - battery.min_soc_mwh
    return {
        "realised_gross_market_revenue_eur": gross,
        "degradation_cost_eur": degradation,
        "realised_net_operating_value_eur": gross - degradation,
        "internal_throughput_mwh": throughput,
        "equivalent_full_cycles": throughput / (2 * usable_range),
    }


def idle_schedule(
    periods: int, battery: BatteryConfig, timestamps: Iterable[object]
) -> pd.DataFrame:
    """Construct a feasible do-nothing schedule for equal initial/terminal SoC."""
    if not np.isclose(battery.initial_soc_mwh, battery.terminal_soc_mwh):
        raise ValueError("idle schedule requires equal initial and terminal state of charge")
    stamp_array = np.asarray(list(timestamps), dtype=object)
    if len(stamp_array) != periods:
        raise ValueError("timestamps must match periods")
    return pd.DataFrame(
        {
            "timestamp": stamp_array,
            "duration_hours": np.ones(periods),
            "decision_price_eur_per_mwh": np.zeros(periods),
            "charge_mw": np.zeros(periods),
            "discharge_mw": np.zeros(periods),
            "soc_start_mwh": np.full(periods, battery.initial_soc_mwh),
            "soc_end_mwh": np.full(periods, battery.initial_soc_mwh),
            "charge_mode": np.zeros(periods, dtype=int),
            "internal_throughput_mwh": np.zeros(periods),
        }
    )
