from __future__ import annotations

import numpy as np
import pytest

from power_market.battery import OptimisationError, evaluate_schedule, optimise_schedule


def _battery(settings, **updates):
    defaults = {
        "energy_capacity_mwh": 1.0,
        "max_charge_power_mw": 1.0,
        "max_discharge_power_mw": 1.0,
        "min_soc_mwh": 0.0,
        "max_soc_mwh": 1.0,
        "initial_soc_mwh": 0.0,
        "terminal_soc_mwh": 0.0,
        "charge_efficiency": 1.0,
        "discharge_efficiency": 1.0,
        "degradation_cost_eur_per_mwh": 0.0,
    }
    defaults.update(updates)
    return settings.battery.model_copy(update=defaults)


def test_known_two_period_optimum_and_realised_value(settings) -> None:
    battery = _battery(settings)
    result = optimise_schedule([0, 100], battery)
    assert result.schedule["charge_mw"].tolist() == pytest.approx([1, 0], abs=1e-7)
    assert result.schedule["discharge_mw"].tolist() == pytest.approx([0, 1], abs=1e-7)
    value = evaluate_schedule(result.schedule, [0, 100], battery)
    assert value["realised_gross_market_revenue_eur"] == pytest.approx(100)
    assert value["realised_net_operating_value_eur"] == pytest.approx(result.forecast_net_value_eur)


def test_separate_efficiencies_and_soc_equation(settings) -> None:
    battery = _battery(settings, charge_efficiency=0.8, discharge_efficiency=0.9)
    schedule = optimise_schedule([0, 100], battery).schedule
    assert schedule.loc[0, "soc_end_mwh"] == pytest.approx(0.8)
    assert schedule.loc[1, "discharge_mw"] == pytest.approx(0.72)
    assert schedule["internal_throughput_mwh"].sum() == pytest.approx(1.6)


def test_negative_prices_preserved_without_simultaneous_cycle(settings) -> None:
    battery = _battery(settings, charge_efficiency=0.8, discharge_efficiency=0.8)
    single = optimise_schedule([-100], battery).schedule
    assert single["charge_mw"].iloc[0] == pytest.approx(0, abs=1e-7)
    assert single["discharge_mw"].iloc[0] == pytest.approx(0, abs=1e-7)
    cycle = optimise_schedule([-50, 100], _battery(settings)).schedule
    assert cycle["charge_mw"].iloc[0] == pytest.approx(1)
    assert cycle["discharge_mw"].iloc[1] == pytest.approx(1)
    assert not ((cycle["charge_mw"] > 1e-8) & (cycle["discharge_mw"] > 1e-8)).any()


def test_degradation_can_make_cycle_uneconomic(settings) -> None:
    battery = _battery(settings, degradation_cost_eur_per_mwh=6.0)
    schedule = optimise_schedule([0, 10], battery).schedule
    assert schedule[["charge_mw", "discharge_mw"]].to_numpy().max() == pytest.approx(0)


def test_realised_revenue_uses_actual_not_forecast(settings) -> None:
    battery = _battery(settings)
    schedule = optimise_schedule([0, 100], battery).schedule
    value = evaluate_schedule(schedule, [80, 20], battery)
    assert value["realised_gross_market_revenue_eur"] == pytest.approx(-60)


def test_duration_scales_energy_and_revenue(settings) -> None:
    battery = _battery(settings, energy_capacity_mwh=0.5, max_soc_mwh=0.5)
    schedule = optimise_schedule([0, 100], battery, durations_hours=[0.5, 0.5]).schedule
    assert schedule.loc[0, "soc_end_mwh"] == pytest.approx(0.5)
    assert evaluate_schedule(schedule, [0, 100], battery)[
        "realised_gross_market_revenue_eur"
    ] == pytest.approx(50)


@pytest.mark.parametrize("periods", [23, 25])
def test_dst_length_horizons_do_not_assume_24(settings, periods: int) -> None:
    battery = _battery(settings)
    result = optimise_schedule(np.sin(np.arange(periods)), battery)
    assert len(result.schedule) == periods
    assert result.schedule["soc_end_mwh"].iloc[-1] == pytest.approx(0)


def test_unreachable_terminal_is_infeasible(settings) -> None:
    battery = _battery(settings, terminal_soc_mwh=1.0, max_charge_power_mw=0.1)
    with pytest.raises(OptimisationError, match="optimum"):
        optimise_schedule([10], battery)


def test_invalid_inputs_rejected(settings) -> None:
    with pytest.raises(ValueError, match="finite"):
        optimise_schedule([np.nan], _battery(settings))
    with pytest.raises(ValueError, match="positive"):
        optimise_schedule([1], _battery(settings), durations_hours=[0])
    schedule = optimise_schedule([1, 2], _battery(settings)).schedule
    with pytest.raises(ValueError, match="match"):
        evaluate_schedule(schedule, [1], _battery(settings))


def test_perfect_foresight_upper_bound(settings) -> None:
    battery = _battery(settings, degradation_cost_eur_per_mwh=1)
    actual = np.array([-5, 20, 0, 40], dtype=float)
    perfect = optimise_schedule(actual, battery).schedule
    poor = optimise_schedule(actual[::-1], battery).schedule
    perfect_value = evaluate_schedule(perfect, actual, battery)["realised_net_operating_value_eur"]
    poor_value = evaluate_schedule(poor, actual, battery)["realised_net_operating_value_eur"]
    assert perfect_value >= poor_value - 1e-7
