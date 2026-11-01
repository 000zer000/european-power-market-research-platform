# Independent battery audit — 1 November 2026

## Reproduced defect and correction

The former auditor accepted an externally supplied three-interval schedule that
jumped from zero to one MWh between rows without charging. At 100 EUR/MWh it
reported 98 EUR after degradation. Replacing its stored throughput with zero
reported 100 EUR. Both regression tests failed before the correction and pass
after it. This did not demonstrate a defective MILP: its state vector was already
chained. The defect was in independent verification of external schedules.

The auditor now requires finite physical values, positive durations, both bounds
on every start/end state, inter-row continuity, endpoints, energy dynamics,
exclusive modes and consistent optional binary mode labels. Tolerances are
absolute (default 1e-6 MW/MWh), with no scale-dependent relative allowance.
Throughput is reconstructed from durations, power and separate efficiencies;
inconsistent stored throughput is rejected. Accounting uses reconstructed flows.
Datetime interval starts must be contiguous in UTC with declared duration
(default timestamp tolerance 1e-9 hours); numeric starts are ordered ordinal
identifiers, not a physical clock. Indexed datetime settlement prices must align.
Unindexed vectors are positional and cannot establish their own time provenance.

## Verification performed

- Python 3.12.14, pinned project direct dependencies, native SciPy/HiGHS.
- 81 tests passed, zero skipped; 90.98% measured branch coverage.
- Ruff check/format and strict source Mypy passed.
- Offline complete pipeline ran inside integration tests, including its separate
  stages and outputs, using temporary directories.
- The committed-evidence contract independently audited all 2,555 daily schedules
  (seven strategies, 365 local delivery days; 61,320 intervals) and reconciled
  five physical/accounting metrics per day against the published daily table.
  Settlement prices were checked against forecast-table UTC timestamps.
- All existing committed evidence passed unchanged, including 23/25-hour days.
  No historical files or totals were replaced. A fresh full official-data model
  retraining and sensitivity rerun were not performed for this validator fix.

Run `python tools/validate_research_artifacts.py` after installing the project.
The CI evidence-contract job executes this enhanced check on future changes.
The original trading-cost, daily terminal-state and historical-simulation limits
remain. No simulated value is asserted to be real trading profit.
