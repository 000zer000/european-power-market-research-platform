# Research artifact contract

The repository commits selected forecasting, validation, and battery-backtest
outputs so that published claims can be inspected without rerunning the full
data acquisition and modelling pipeline. The artifact contract prevents those
files from silently drifting out of agreement.

## Automated checks

Run:

```bash
python tools/validate_research_artifacts.py
```

The validator checks the following cross-file invariants:

- all core evidence files are present and valid JSON or CSV;
- the final prediction file contains a continuous, unique 8,760-hour UTC test
  series for 2025;
- required forecast columns are complete and contain no missing values;
- published MAE and RMSE values are reproduced from the committed predictions;
- every battery strategy covers the same 365 local delivery days;
- battery net operating value equals gross market revenue minus degradation
  cost;
- degradation, throughput, cycles, and perfect-foresight gaps respect their
  physical or accounting bounds;
- the perfect-foresight strategy has a zero value gap;
- prediction and battery-backtest date coverage agree;
- published daily forecast-error versus value-gap correlations are reproduced.

CI runs the contract independently of the unit-test matrix. A pull request that
changes predictions, metrics, or backtest outputs must update all dependent
evidence consistently.

## What this does not prove

Passing the contract does not independently validate the economic assumptions,
forecasting methodology, source data, or causal interpretation. It verifies
that the **committed evidence is internally consistent** with the repository's
published calculations.

The source-data provenance manifest contains hashes for files intentionally not
committed to Git. Those hashes support local source verification, but CI cannot
recompute them without downloading the raw SMARD inputs. The artifact contract
therefore confirms that the manifest exists and is valid JSON; it does not
claim to re-download or authenticate upstream data during every pull request.

## Expected maintenance workflow

1. Rerun the documented offline or online pipeline.
2. Review the generated metrics, predictions, and battery results.
3. Run the normal unit tests and quality checks.
4. Run `python tools/validate_research_artifacts.py`.
5. Commit dependent outputs together, with an explanation of why the research
   result changed.

Never edit a headline metric by hand without updating and validating the
underlying prediction or backtest evidence.
