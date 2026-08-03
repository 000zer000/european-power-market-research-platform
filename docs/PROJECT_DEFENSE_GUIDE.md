# Project defense guide

This guide is the material to learn, not a script to recite without understanding. Open the named function while studying each section and run the corresponding test when possible.

## 1. Beginner overview

Electricity for tomorrow is bought and sold today. The auction produces a price for every delivery interval. A battery can buy energy when the price is low and sell it later when the price is high, but it loses energy, has limited power and capacity, and wears as it moves energy.

This project asks a precise question: if a model forecasts tomorrow's DE-LU prices without seeing future information, does the schedule chosen from that forecast earn more when settled at actual prices than schedules based on yesterday or last week?

There are three different numbers you must never confuse:

1. Forecast accuracy: how close a predicted price is to the actual price.
2. Gross market revenue: actual sale proceeds minus actual purchase cost.
3. Modeled net operating value: gross revenue minus the configured throughput degradation proxy.

The project does not calculate accounting profit or return on investment.

## 2. Market terminology

- Bidding zone: an area with one wholesale price in the coupled auction. This project uses Germany/Luxembourg, `DE-LU`.
- Day-ahead market: today's auction for energy delivered tomorrow.
- Delivery interval / market time unit: the period to which one auction product and price applies. It was hourly for most of the sample and changed to 15 minutes in October 2025; this hourly study uses SMARD's official hourly series.
- EUR/MWh: euros paid for one megawatt-hour of energy. It is not power.
- MW: instantaneous charge or discharge power.
- MWh: energy, equal to MW multiplied by hours.
- Negative price: sellers pay buyers to consume energy. It is legitimate and must not be clipped.
- State of charge (SoC): energy currently stored, measured here in MWh.
- Charge/discharge efficiency: fractions representing one-way conversion losses.
- Round-trip efficiency: charge efficiency multiplied by discharge efficiency.
- Throughput: energy processed by the cell; used as a simple degradation proxy.
- Arbitrage: moving energy from a lower-price interval to a higher-price interval.
- Price taker: an actor assumed too small to change the market price.
- Perfect foresight: a non-deployable scheduler allowed to know actual prices, used only as an upper bound.
- Market value versus accounting profit: market value here includes settled energy revenue and the modeled degradation charge; profit would also need capex, fees, tax, O&M and other omitted costs.

## 3. Complete data flow

1. `load_settings` reads YAML and Pydantic rejects unknown fields and impossible battery bounds.
2. `fetch_smard` downloads the official weekly JSON index and only the chunks that overlap 2023–2025. A byte-for-byte cache avoids repeated traffic.
3. `parse_smard_payload` accepts exactly the expected top-level schema and converts Unix milliseconds and numeric prices without touching negative values.
4. `_canonicalise` filters the requested local-date range after converting timestamps to UTC.
5. `fetch_prices` writes the canonical CSV and a manifest containing URL, retrieval time, SHA-256, unit and licence.
6. `validate_prices` reconstructs every expected UTC hour and checks schema, timestamps, values, duplicates, missing intervals, extra intervals and local DST day lengths.
7. `build_features` adds local calendar variables and price lags/rolling statistics that end at least 24 hours before the target.
8. `feature_availability_audit` reconstructs exact lag and rolling values. If the audit fails, features are not saved.
9. `select_models` fits candidate settings on 2023 and ranks them on 2024.
10. `walk_forward_predict` freezes the selected settings and produces 2025 forecasts with monthly expanding retraining.
11. `evaluate_forecasts` reports overall and sliced errors plus day-block confidence intervals.
12. `_run_day` hands each strategy's predicted prices—not actual prices—to the battery optimizer.
13. `evaluate_schedule` values the fixed schedule using actual prices and subtracts degradation.
14. `run_backtest` asserts the perfect-foresight upper bound and saves interval, daily and aggregate evidence.
15. `reporting.py` turns saved evidence into figures and a measured summary.

At no stage is a bad source observation silently filled. At no stage does an ordinary forecast-driven optimizer receive actual future prices.

## 4. Models and baselines

### Previous-day baseline

Prediction for timestamp \(t\) is the price at \(t-24\) elapsed hours. It is strong because power prices repeat daily patterns. Across DST it is deliberately elapsed-time based, so it stays unambiguous in UTC.

### Previous-week baseline

Prediction is the price at \(t-168\) hours. It can preserve weekday-specific behavior that a one-day lag misses.

### Ridge regression

Ridge minimizes squared error plus \(\alpha\sum_j\beta_j^2\). The penalty shrinks correlated coefficients and reduces variance. Standardization is essential because hour, prices and sine/cosine features have different scales. Alpha is selected on 2024, never 2025.

### Histogram gradient boosting

The model adds shallow decision trees sequentially to correct prior errors. It captures nonlinear interactions—for example, a previous-day price may mean something different at 18:00 in winter than at noon in summer. Histogram binning makes training efficient. Learning rate and maximum leaves are selected on 2024. It is still a tabular model, not a time-series oracle; temporal validity comes from feature construction and splitting.

## 5. Leakage: what it is and why it matters

Leakage means the model receives information that would not exist at decision time. It makes an experiment look better than a deployable system.

Examples this project prevents:

- random train/test splitting, which lets later market regimes teach earlier rows;
- a rolling mean containing the current target;
- normalizing on the full 2025 test distribution;
- selecting hyperparameters after comparing 2025 MAE;
- scheduling on actual prices and calling the result “forecast-driven”;
- resolving duplicate autumn local times by dropping one row.

The key line in `features.py` is `safe_history = prices_series.shift(24)`. All rolling price statistics start there. The tests compare feature columns to exact source positions and mutate future targets to prove earlier features do not move.

## 6. Walk-forward validation

A static 2023 model could become stale. A random split would be invalid. The compromise is:

- select model settings once using 2023 → 2024;
- refit with all data available before January 2025 and predict January;
- after January actuals exist, add them to history, refit and predict February;
- continue through December.

This is an expanding window because history grows. It is “walk-forward” because the simulated origin moves forward in calendar order. The 2025 outcomes can update later 2025 fits only after those outcomes would have become historical; they never choose hyperparameters.

## 7. Battery formulation with units

Let charge and discharge be grid-meter power in MW and interval duration be hours.

\[
s_{t+1}\,[MWh]=s_t\,[MWh]+\eta_c c_t\,[MW]\Delta_t\,[h]-\frac{d_t\,[MW]}{\eta_d}\Delta_t\,[h].
\]

If \(\eta_c=0.95\), charging at 1 MW for one hour adds 0.95 MWh. If \(\eta_d=0.95\), exporting 0.95 MWh requires removing 1 MWh.

The binary \(y_t\) enforces:

\[
c_t\le P_cy_t,\quad d_t\le P_d(1-y_t).
\]

If \(y_t=1\), discharge must be zero. If \(y_t=0\), charge must be zero. This is necessary under negative prices.

The optimizer maximizes forecast-valued market revenue minus internal-throughput cost. After the decision, revenue is recalculated using actual prices. Daily terminal SoC equals initial SoC so every strategy begins the next comparison with the same energy.

## 8. Why accuracy and value can disagree

MAE weights every hour equally. A battery cannot act on every small error because power and energy are limited. What matters is the price ordering and spread around the hours selected for charging and discharging.

A model can reduce many EUR 2/MWh errors during inactive hours yet reverse the ranking of the day's cheapest two hours; MAE improves while battery value falls. Conversely, a forecast can have mediocre MAE but correctly identify the extreme low and high windows, producing good value. Degradation can also make a small predicted spread uneconomic.

Use `forecast_to_value.json` carefully. Its daily correlation describes association between daily MAE and the gap to perfect foresight; it does not prove that reducing MAE by one unit causes a particular revenue increase.

## 9. How to read every chart

### Executed result you must be able to explain

Ridge achieved the lowest 2025 MAE at 23.375 EUR/MWh, a 9.98% reduction from previous day's 25.965. Its daily-bootstrap difference interval did not cross zero. Histogram gradient boosting had slightly worse MAE at 24.380, but its forecast-driven schedule produced the highest modeled net operating value, EUR 71,801.98, versus EUR 70,761.02 for Ridge and EUR 67,934.64 for previous day. Perfect foresight was EUR 82,114.29. Your explanation is: MAE weights all delivery hours equally, while a finite battery earns value from the ordering and spread of the few intervals it can act on. Do not say the tree model was “more accurate”; it was less accurate by MAE but better for this downstream decision.

### Price distribution and negative prices

The histogram is clipped only for display at the 0.5th and 99.5th percentiles; calculations use every price. The zero line and annual negative-price share show why a scheduler must accept negative values and why a binary exclusivity constraint is not optional.

### Actual versus forecast

This is an untouched 2025 week. Compare timing of peaks and troughs, not only vertical distance. A battery reacts to relative ordering.

### Model error comparison

MAE is typical absolute miss; RMSE increases more when extreme misses occur. The chart is descriptive final-test evidence. Model selection was completed before these bars were calculated.

### Error by hour

The x-axis is Europe/Berlin delivery hour. Higher evening or morning errors suggest systematic difficulty around ramps and demand peaks. These slices diagnose behavior; they were not used for post-test tuning.

### Battery dispatch and SoC

Negative bars mean grid charging, positive bars mean discharge. SoC rises by only efficiency-adjusted charge energy and falls by more than exported energy. The chosen example includes a negative-price day when available.

### Cumulative realised value

The final point equals each strategy's aggregate net operating value, but the path shows drawdowns and regime dependence. Perfect foresight should remain an upper-bound comparator in aggregate; crossing paths on individual partial periods can occur only if comparing different endpoint groupings, so investigate any surprising plot.

### Perfect-foresight gap

This is upper-bound net value minus strategy net value under identical constraints. A smaller gap is better. It measures both forecast and decision loss within the simplified formulation.

### Sensitivity

Perfect foresight is held fixed as the information assumption while efficiency, power/energy ratio or degradation cost changes. This isolates physical/economic assumptions. It does not predict an investable project's profit.

## 10. Codebase map

| Question | Start here | Proof test |
|---|---|---|
| How is config validated? | `src/power_market/config.py` | `tests/test_config.py` |
| How is official JSON parsed? | `parse_smard_payload` | `tests/test_data.py` |
| How are DST hours retained? | `_date_bounds_utc`, `validate_prices` | `tests/test_validation.py` |
| Where is leakage blocked? | `build_features` | `tests/test_features.py` |
| Where are splits enforced? | `assign_split`, `select_models` | `tests/test_forecasting.py` |
| How are metrics defined? | `src/power_market/metrics.py` | `tests/test_metrics.py` |
| Where is the MILP built? | `optimise_schedule` | `tests/test_battery.py` |
| How is actual revenue separated? | `evaluate_schedule`, `_run_day` | realised-revenue test |
| How are plots reproduced? | `src/power_market/reporting.py` | integration test |
| How does everything run? | `src/power_market/pipeline.py` | `tests/test_integration.py` |

## 11. Five-minute interview presentation

“I built a Python research platform to test whether leakage-free day-ahead price forecasts create battery scheduling value in the German-Luxembourg market. I used official Bundesnetzagentur SMARD prices from 2023 through 2025, because the preferred Fraunhofer Energy-Charts API was unavailable during the reproducibility run. Every raw response is cached and hashed, timestamps are unique UTC keys, and validation explicitly proves the 23- and 25-hour DST days and preserves negative prices.

The forecasting experiment trains on 2023, selects Ridge and histogram gradient boosting hyperparameters on 2024, and leaves 2025 untouched for final evaluation. Price-derived features are shifted at least 24 hours, rolling windows exclude the target, and automated mutation tests catch future leakage. In 2025 I retrain monthly on an expanding window and compare both fitted models with previous-day and previous-week seasonal baselines using MAE, RMSE, directional accuracy, hourly/monthly slices and paired day-block bootstrap intervals.

For economic evaluation I wrote a daily mixed-integer battery optimizer with separate efficiencies, state-of-charge dynamics, power and energy limits, equal terminal SoC and a throughput degradation charge. A binary variable prevents simultaneous charge and discharge, which matters when prices are negative. Each strategy schedules only on its predicted prices; a separate function settles that frozen schedule against actual prices. Perfect foresight uses the same constraints as a verified upper bound.

The main lesson is that forecast accuracy and storage value are related but not identical: value depends on ranking the constrained charge and discharge hours. I report gross market revenue, degradation and modeled net operating value separately and explicitly exclude capex, fees, imbalance and bidding rules. The repository is installable, typed, tested, linted and reproducible from one command.”

Then state the exact winning model, MAE and battery values from `reports/research_summary.md`; do not memorize numbers from an older run.

## 12. Fifteen-minute technical presentation

Use this order and spend roughly the indicated time:

1. Problem and hypothesis (1 minute): accuracy alone is not the final objective; forecast-driven storage value is.
2. Data contract (2 minutes): official source, CC BY attribution, UTC key, Europe/Berlin features, 23/25-hour days, October 2025 resolution caveat, validation report.
3. Decision-time information (2 minutes): auction timing, calendar availability, exact 24/168 lags, shifted rolling windows, mutation test.
4. Experimental design (2 minutes): 2023/2024/2025 roles, validation grid, monthly expanding refit, deterministic seed, no random split.
5. Forecast results (2 minutes): MAE/RMSE ranking, bootstrap interval, negative/extreme slices and hardest hours. Say only what the artifacts show.
6. Battery mathematics (3 minutes): units, SoC equation, binary exclusivity, terminal equality, degradation objective, independent solver audit.
7. Economic results (2 minutes): realised gross/cost/net, best forecast-driven strategy, time-of-use and seasonal comparisons, perfect gap, cycles, sensitivity.
8. Limitations and next experiment (1 minute): quarter-hour modeling after October 2025, fundamentals with availability timestamps, cross-day SoC and actual market rules.

Have `data_validation.json`, `selected_models.json`, `forecast_metrics.json` and `battery_summary.csv` open. They are stronger evidence than slides.

## 13. Interview questions and accurate answers

1. **Why is random splitting wrong?** Adjacent electricity prices are serially dependent and regimes change. Random rows let future patterns influence training and overstate deployable performance.
2. **Why is 2025 untouched during selection?** It provides an honest final estimate after all feature and hyperparameter decisions are frozen on earlier periods.
3. **Does using January 2025 to train February leak?** No. In a walk-forward simulation January outcomes are historical by February. They cannot choose hyperparameters, which remain fixed from 2024.
4. **Why use UTC as the key?** UTC is unique and continuous. Autumn local time repeats 02:00; a naïve local timestamp would collide.
5. **How do you preserve local behavior?** Convert the UTC index to Europe/Berlin only for calendar features and daily grouping, retaining the UTC offset.
6. **What happens on spring DST?** The local day has 23 real hourly intervals. The optimizer accepts an arbitrary horizon length and still enforces its terminal state.
7. **What happens on autumn DST?** It has 25 UTC hours; both local 02:00 intervals remain distinct by offset.
8. **Why not interpolate a missing price?** That invents an auction outcome and hides a reproducibility problem. Validation fails and reports the exact missing timestamp.
9. **Why are negative prices valid?** Oversupply and inflexible production can make market participants willing to pay for consumption. Clipping them destroys market meaning and battery incentives.
10. **Why previous-day and previous-week baselines?** Electricity prices have strong daily and weekly seasonality; a fitted model must earn its complexity against credible simple alternatives.
11. **Why Ridge?** It is interpretable, fast and stabilizes correlated lag/calendar features through L2 shrinkage.
12. **Why histogram gradient boosting?** It captures nonlinear thresholds and interactions with a stable mainstream dependency and modest compute.
13. **Why standardize Ridge but not trees?** Ridge's coefficient penalty depends on scale; tree split ordering is invariant to monotonic scaling.
14. **What is the directional metric?** Whether forecast and actual move in the same direction relative to the previous-day price for that interval, excluding actual ties.
15. **Why bootstrap days rather than hours?** Errors within a day are dependent. Resampling 24-hour blocks better preserves that structure.
16. **Does a 95% interval prove superiority?** No. It quantifies sampling uncertainty under the chosen day-resampling scheme; it does not cover regime change or model specification risk.
17. **Why can lower MAE produce lower value?** MAE weights inactive and decisive hours equally, while the battery cares about constrained price ordering and spreads.
18. **What makes the battery optimization mixed integer?** A binary mode per interval chooses charge or discharge and prevents both being positive.
19. **Why not a relaxed linear model?** At negative prices, simultaneous lossy charging/discharging can become artificially profitable while honoring net SoC.
20. **Explain the SoC equation.** Stored energy rises by charge power times duration times charge efficiency and falls by discharge power times duration divided by discharge efficiency.
21. **Why divide discharge by efficiency?** To export one MWh when efficiency is below one, more than one MWh must leave the cell.
22. **Why equal initial and terminal SoC daily?** It removes free terminal energy and makes daily strategies comparable, though it excludes cross-midnight arbitrage.
23. **How is degradation modeled?** A linear euro charge on internal-cell throughput, charge-side stored energy plus discharge-side removed energy. It is a proxy, not electrochemistry.
24. **What is an equivalent full cycle?** Total internal throughput divided by twice the usable energy range.
25. **How do you prove actual prices value the schedule?** Optimization and evaluation are separate functions; a controlled test schedules on `[0,100]`, evaluates on `[80,20]`, and verifies a EUR -60 gross result.
26. **Why is perfect foresight an upper bound?** It optimizes the exact realised objective over the same feasible set. No schedule chosen using less information can exceed its optimum beyond tolerance.
27. **Could perfect foresight lose money?** With equal endpoints and optional idle operation, its optimum cannot be below zero modeled net value; it can always do nothing.
28. **Is gross revenue profit?** No. Even net operating value here excludes capex, fixed costs, fees, imbalance, taxes and operational constraints.
29. **What does the provenance hash prove?** It proves which exact response bytes produced the run and detects later cache changes. It does not prove the provider's data are economically correct.
30. **Why did you use SMARD rather than Energy-Charts?** The documented preferred API persistently returned 503 during execution; SMARD is the authoritative Bundesnetzagentur source, live, CC BY 4.0 and uses the same underlying DE-LU series.
31. **What changed in October 2025?** European day-ahead products moved to 15-minute market time units. This study uses SMARD's official hourly series for a uniform panel and states that intrahour value is excluded.
32. **How is model determinism tested?** Fixed seeds, deterministic candidate grids, disabled HGB early stopping, repeated prediction runs and exact frame comparison.
33. **What could you add next without leakage?** Load, wind and solar forecasts stamped with their publication/issue times, never realised fundamentals for the delivery interval.
34. **What is the most important production gap?** Market participation and settlement rules: bid granularity, gate closure, fees, imbalance and operational reliability.
35. **What claim would you refuse to make?** That the modeled annual value is accounting profit or that the best 2025 model will remain best in future regimes.

## 14. Code-reading and modification exercises

### Exercise 1: locate the leakage boundary

Question: In `build_features`, change `safe_history = prices_series.shift(24)` to `prices_series`. Which test should fail and why?

Solution: `test_features_use_exact_past_prices` or the feature availability audit fails because the 168-hour rolling mean would include observations less than 24 hours before the target. More importantly, the feature would violate the declared next-day information set.

### Exercise 2: calculate one transition

Question: Initial SoC is 1 MWh, charge is 1 MW for one hour, \(\eta_c=0.95\), discharge is zero. What is final SoC?

Solution: \(1+0.95\times1\times1=1.95\) MWh, provided the maximum SoC permits it.

### Exercise 3: catch an efficiency bug

Question: A developer changes the discharge transition to `- discharge * efficiency`. Why is it wrong?

Solution: It says exporting 1 MWh removes less than 1 MWh from the cell, creating energy. Grid-side discharge must be divided by discharge efficiency. `test_separate_efficiencies_and_soc_equation` catches it.

### Exercise 4: prove realised valuation separation

Question: Add a forecast `[0,100]` and actual `[80,20]` with a perfect one-MWh battery. What should gross value be?

Solution: The forecast schedules charge first and discharge second. Realised cash flow is `-80 + 20 = -60 EUR`. If the code returns `100`, it reused forecast prices during valuation.

### Exercise 5: add a maximum cycle constraint

Question: How would you limit daily internal throughput to 4 MWh?

Solution: Add one linear inequality whose charge coefficients are \(\eta_c\Delta_t\), discharge coefficients are \(\Delta_t/\eta_d\), and upper bound is 4. Add a controlled test where unconstrained throughput exceeds four and assert the new total and perfect-foresight property.

### Exercise 6: add a new known-in-advance feature

Question: Add “day of year” cyclic encoding safely.

Solution: Derive local day-of-year from `timestamp_utc.tz_convert(Europe/Berlin)`, add sine/cosine with a 365.25 denominator, append both names to `FEATURE_COLUMNS`, document them as calendar-known, and rerun leakage, deterministic and final tests. Do not tune the feature after inspecting 2025 results.

### Exercise 7: debug a 25-hour duplicate error

Question: A merge drops one autumn 02:00 row. What key was probably used?

Solution: A naïve local timestamp or `(local_date, local_hour)`. Use the timezone-aware UTC timestamp as the unique key and retain local offset only as metadata.

### Exercise 8: interpret a negative price

Question: At -50 EUR/MWh, a battery charges 1 MW for one hour. What is gross market cash flow?

Solution: Revenue formula is \(p(d-c)\Delta\). With zero discharge it is \((-50)(0-1)(1)=+50\) EUR: the battery is paid to consume. SoC increases by efficiency-adjusted energy.

### Exercise 9: change retraining frequency

Question: What must change to retrain quarterly?

Solution: Group test rows by quarter, fit history strictly before the first UTC timestamp of each quarter, update the validated config literal, document the new cadence and add a test proving every training timestamp precedes the prediction origin. Do not compare quarterly and monthly using 2025 to choose one without a new nested design.

### Exercise 10: investigate a perfect-foresight violation

Question: A forecast strategy exceeds perfect foresight by EUR 5 on one day. What do you check first?

Solution: Confirm both use identical actual valuation prices, endpoints, efficiencies, power/energy bounds, durations and degradation definition; confirm the perfect solver reached status 0; reconstruct both objectives; then inspect timestamp alignment. Do not clip the gap to zero.

## 15. Study order

1. Sections 1–3 and the architecture diagram.
2. Run the offline fixture and inspect `data_validation.json`.
3. Study Sections 5–6 beside `features.py` and `test_features.py`.
4. Study Section 7 beside `battery.py` and work Exercises 2–4 by hand.
5. Read the executed forecast and battery artifact tables, then Section 9.
6. Learn the five-minute presentation in your own words.
7. Answer questions 1–30 without the guide.
8. Complete exercises 5–10 in a temporary branch.
9. Practice the fifteen-minute presentation with the four machine-readable evidence files open.
