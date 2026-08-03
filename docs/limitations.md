# Honest limitations

This repository answers a controlled research question; it is not a deployable trading system or an investment case.

- The 2025-10-01 European day-ahead market-time-unit change is represented by SMARD’s official hourly series. Intrahour quarter-hour dispatch and value are not modeled.
- Forecasts use price and calendar history only. Known-in-advance load, wind, solar, fuel, carbon, outage and interconnector forecasts could improve accuracy but require separate availability audits.
- UTC 24/168-hour lags avoid ambiguous DST labels but are not always the same local clock hour across transition days.
- Monthly retraining is a reproducible research convention, not an optimized production cadence.
- The battery has perfect availability and price-taking execution. There are no minimum bid sizes, exchange order constraints, partial fills, imbalance settlement, market-access rules, latency, downtime or forecast publication delays.
- Capex, fixed O&M, market fees, taxes, financing, warranty conditions, thermal limits, calendar ageing and nonlinear state-dependent degradation are excluded.
- The degradation charge is a configurable linear proxy on internal-cell throughput, not an electrochemical ageing model.
- Daily initial and terminal state-of-charge equality makes strategy comparisons clean but can prevent economically useful energy transfer across midnight.
- Perfect foresight is an information upper bound under the simplified formulation, not attainable performance.
- Bootstrap intervals resample days and quantify sampling uncertainty in paired MAE only. They do not capture structural regime change or data revisions.
- One bidding zone and three historical years are insufficient for claims about all European markets or future regimes.
- SMARD warns that it does not guarantee completeness or correctness even though Bundesnetzagentur checks the ENTSO-E-fed data.

Claims not supported by this project:

- that the chosen model will remain best after 2025;
- that forecast MAE improvement guarantees higher battery value;
- that modeled net operating value is accounting profit or return on investment;
- that perfect-foresight value is achievable;
- that the code is ready for live bidding, risk limits or production operations;
- that the results apply outside DE-LU.
