# Results and evidence map

The measured values in this repository are generated—not hand-entered—from the same saved predictions and schedules used by the tests.

## Executed 2025 results

Ridge had the lowest price error: MAE **23.375 EUR/MWh** and RMSE **33.852 EUR/MWh**, compared with **25.965** and **40.446** for the strongest seasonal MAE baseline, previous day. Its paired daily MAE difference from previous day was **-2.595 EUR/MWh**, with a 95% day-block-bootstrap interval **[-3.822, -1.408]**. Histogram gradient boosting had MAE **24.380** and interval **[-3.233, -0.071]** versus previous day. On negative-price hours alone, the previous-day baseline was strongest; this is why the repository does not claim universal ML dominance.

The value ranking differed. Histogram gradient boosting produced the highest forecast-driven modeled net operating value at **EUR 71,801.98**, followed by Ridge at **EUR 70,761.02**, previous week at **EUR 69,720.75**, and previous day at **EUR 67,934.64**. Perfect foresight was **EUR 82,114.29**. Thus the nonlinear model captured **87.44%** of the simplified upper bound even though Ridge had lower MAE. This is the core research finding: average price accuracy and constrained decision value are not interchangeable.

The default sensitivity upper bound fell from **EUR 82,114.29** to **EUR 58,442.48** when each one-way efficiency changed from 0.95 to 0.85, rose to **EUR 90,126.26** when the power/capacity ratio rose from 0.5 to 1.0 per hour, and fell to **EUR 63,412.08** when degradation increased from EUR 2 to EUR 10 per MWh of internal throughput. These are modeled operating values, not project profits.

| Evidence | File | What to verify |
|---|---|---|
| Data quality | `artifacts/validation/data_validation.json` | pass status, hourly continuity, negative prices, 23/25-hour days |
| Raw provenance | `artifacts/provenance/manifest.json` | official URLs, SHA-256, timestamps, licence |
| Feature availability | `artifacts/provenance/feature_availability.json` | exact lags, 24-hour minimum price-feature delay |
| Model selection | `artifacts/metrics/validation_model_selection.csv` | candidates ranked on 2024 only |
| Frozen choices | `artifacts/metrics/selected_models.json` | selected parameters and monthly retraining policy |
| Test forecasts | `artifacts/metrics/test_predictions.csv` | every 2025 UTC interval and four methods |
| Forecast metrics | `artifacts/metrics/forecast_metrics.json` | overall/slice metrics and daily bootstrap intervals |
| Hour/month errors | `artifacts/metrics/forecast_errors_by_hour_month.csv` | local-time error profiles |
| Daily battery value | `artifacts/backtests/battery_daily_results.csv` | realised value and perfect gap per local day |
| Dispatch audit trail | `artifacts/backtests/battery_dispatch.csv` | charge, discharge, SoC, throughput and actual price |
| Strategy totals | `artifacts/backtests/battery_summary.csv` | gross revenue, cost, net value, cycles and negative-price days |
| Sensitivity | `artifacts/backtests/battery_sensitivity.csv` | efficiency, power/energy and degradation assumptions |
| Concise narrative | `reports/research_summary.md` | automatically populated measured summary |

Read `forecast_metrics.json` before claiming that a fitted model beat a baseline. Read `battery_summary.csv` before claiming that better price accuracy created more value. Those are related but distinct empirical questions.

The charts embed those same results:

1. `price_distribution.png`: price shape and annual negative-price frequency.
2. `forecast_example.png`: one untouched test week.
3. `model_error_comparison.png`: final-test MAE and RMSE.
4. `error_by_hour.png`: which delivery hours are hardest.
5. `battery_dispatch_example.png`: forecast price, actual price, power and SoC.
6. `cumulative_realised_value.png`: value path rather than only a final total.
7. `perfect_foresight_gap.png`: value left by each implementable strategy.
8. `battery_sensitivity.png`: how simplifying assumptions affect the upper bound.
