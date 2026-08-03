# Executed research summary

Data source: **smard**, DE-LU day-ahead prices, 2023-01-01 through 2025-12-31. The final 2025 delivery year was not used for hyperparameter selection.

## Forecasting

The lowest final-test MAE was **ridge** at **23.375 EUR/MWh** (RMSE **33.852 EUR/MWh**). This is an evaluation result, not a claim that machine learning must outperform seasonal baselines.

## Battery backtest

Among forecast-driven strategies, **hist gradient boosting** produced the highest realised modeled net operating value: **EUR 71801.98** over **365** days. Perfect foresight produced **EUR 82114.29**, an upper bound under the same physical and cost assumptions.

“Net operating value” is realised day-ahead market revenue minus the configured internal-throughput degradation charge. It is not accounting profit: capex, market-access fees, imbalance costs, bidding restrictions, taxes and operational risk are excluded.
