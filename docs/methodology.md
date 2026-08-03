# Forecasting methodology

## Research question

Can day-ahead forecasts chosen without test leakage improve the realised modeled value of a battery relative to seasonal baselines? This is deliberately two linked experiments: price accuracy and forecast-driven decisions. A lower MAE need not create more storage value because timing errors around the day's cheapest and most expensive hours matter more than many small errors elsewhere.

## Information set

The forecast represents the full next delivery day. Calendar variables are known in advance. Prior day-ahead auction prices are public before the next auction decision, so prices from the previous delivery day are available even where physical delivery has not yet completed. The implementation remains conservative: every price-derived feature is shifted by at least 24 elapsed delivery hours.

| Feature family | Exact implementation | Earliest source relative to target | Why available |
|---|---|---:|---|
| Calendar | local hour, weekday, month, weekend | not data-dependent | calendar is known |
| Cyclical calendar | sine/cosine of hour, weekday, month | not data-dependent | deterministic transform |
| Holiday | nationwide German public holiday | not data-dependent | dates known in advance |
| Previous day | UTC lag 24 | 24 hours | earlier auction result |
| Previous week | UTC lag 168 | 168 hours | earlier auction result |
| Daily history | 24-hour mean after a 24-hour shift | 24–47 hours | earlier prices only |
| Weekly history | 168-hour mean/std after a 24-hour shift | 24–191 hours | earlier prices only |

UTC lags are unambiguous across DST. On the two transition days they are elapsed-hour rather than “same clock label yesterday” lags. This avoids ambiguous duplicate local times and is an explicit modeling choice.

## Leakage controls

1. Rows are sorted and keyed by UTC before lags are constructed.
2. Rolling windows operate on `price.shift(24)`, never the unshifted target.
3. The model feature list is fixed and excludes `price_eur_per_mwh`.
4. An availability audit independently reconstructs exact lags and rolling values.
5. Tests mutate future targets and assert that earlier feature rows are unchanged.
6. Split labels use local delivery date and are chronological.
7. Hyperparameters see 2023 training targets and 2024 validation targets only.
8. After selection is frozen, 2025 predictions are produced month by month. Each monthly fit uses only rows earlier than the first timestamp of that month.

The first 191 rows include incomplete rolling history. They remain in the feature file for auditability but are excluded from model fitting; this is a documented feature warm-up, not silent repair of source observations.

## Models

- Previous-day seasonal naïve: the observed day-ahead price 24 elapsed hours earlier.
- Previous-week seasonal naïve: the price 168 elapsed hours earlier.
- Ridge regression: standardized numeric features with L2 shrinkage. Validation chooses alpha from the YAML grid.
- Histogram gradient boosting: nonlinear trees from scikit-learn. Validation chooses learning rate and maximum leaf nodes. A fixed seed and disabled early stopping make repeated runs deterministic.

Ridge is useful because its regularisation controls correlated calendar and lag features. Histogram gradient boosting can represent threshold effects and interactions such as different morning/ evening behavior by season without a heavy deep-learning dependency.

## Splits and retraining

- Train: delivery through 2023-12-31.
- Validation/model selection: 2024-01-01 through 2024-12-31.
- Final test: 2025-01-01 through 2025-12-31.

There is no random split. After validation selection, the test simulation retrains monthly on an expanding window. January 2025 uses data through December 2024; February may use realised January 2025 history; and so on. This imitates a research model updated as observations arrive without using a future test month.

## Metrics

- MAE: average absolute price error, EUR/MWh.
- RMSE: square-root average squared error, EUR/MWh; it gives more weight to large misses.
- Directional accuracy: whether the forecast and actual price move in the same direction relative to the previous-day price for that delivery interval. Exact actual ties are excluded.
- Slices: MAE by local hour, month, negative-price observations, and observations at or beyond the test-set 95th percentile of absolute price. The extreme threshold is used only for evaluation.
- Uncertainty: paired differences in daily MAE are resampled by whole local delivery day. This retains within-day serial dependence better than resampling individual hours. The reported 95% percentile interval uses the deterministic seed.

“Better” is not assigned from a single point estimate. The outputs show seasonal baselines beside both fitted models, and claims follow the executed ranking and confidence interval.
