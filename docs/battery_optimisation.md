# Battery optimisation

## Variables and units

For each one-hour delivery interval \(t\):

- \(\hat p_t\): scheduling price forecast, EUR/MWh;
- \(p_t\): actual settlement price used only after scheduling, EUR/MWh;
- \(c_t,d_t\): grid-side charge and discharge power, MW;
- \(s_t\): energy stored at the start of the interval, MWh;
- \(y_t\in\{0,1\}\): binary mode, one permits charge and zero permits discharge;
- \(\eta_c,\eta_d\): separate one-way efficiencies;
- \(\Delta_t\): interval duration, hours;
- \(\kappa\): degradation charge, EUR per MWh of internal-cell throughput.

The default is 2 MWh energy, 1 MW charge/discharge power, 0–2 MWh allowed state of charge, 1 MWh at both the beginning and end of every local day, 95% charge and discharge efficiency, and EUR 2/MWh internal throughput. The round-trip efficiency is \(0.95^2=90.25\%\), not 95%.

## State equation

\[
s_{t+1}=s_t+\eta_c c_t\Delta_t-\frac{d_t}{\eta_d}\Delta_t.
\]

Charging 1 MWh from the grid stores only \(\eta_c\) MWh. Exporting 1 MWh requires removing \(1/\eta_d\) MWh. This convention prevents impossible energy creation.

Bounds are:

\[
0\le c_t\le P_c,\quad 0\le d_t\le P_d,\quad E_{min}\le s_t\le E_{max}.
\]

Every daily horizon fixes \(s_0=S_{initial}\) and \(s_T=S_{terminal}\). Equal daily endpoints stop a strategy receiving free initial energy or leaving paid energy with no terminal value.

## No simultaneous operation

The tight mixed-integer constraints are:

\[
c_t\le P_c y_t,\qquad d_t\le P_d(1-y_t).
\]

A relaxed linear program could charge and discharge simultaneously when prices are negative, use efficiency losses to consume energy, and receive payment on net imports. The binary variable prevents that artifact. The solver is SciPy `milp` using its HiGHS backend; only mode variables are integral.

## Scheduling objective

Internal throughput is

\[
q_t=(\eta_c c_t+d_t/\eta_d)\Delta_t.
\]

The forecast-time objective maximizes

\[
\hat J=\sum_t\left[\hat p_t(d_t-c_t)\Delta_t-\kappa q_t\right].
\]

SciPy minimizes, so the code negates the economic value in its coefficient vector. After solving, `audit_schedule` independently reconstructs every state equation, bound, endpoint, exclusivity condition and objective. A merely feasible time-limit incumbent is not accepted as a research optimum.

## Realised valuation

The ordinary optimizer receives only a strategy's forecast vector. The chosen fixed schedule is then valued against actual prices:

\[
R^{gross}=\sum_t p_t(d_t-c_t)\Delta_t,
\]

\[
C^{degradation}=\kappa\sum_t q_t,
\]

\[
V^{net}=R^{gross}-C^{degradation}.
\]

Gross market revenue and degradation cost are reported separately. `V_net` is modeled net operating value, not accounting profit. Equivalent full cycles are internal throughput divided by twice the usable energy range.

## Compared strategies

- Do nothing: feasible zero dispatch.
- Time of use: fixed off-peak hours 00:00–05:00 and peak hours 17:00–20:00 create a known-in-advance scheduling signal.
- Previous day, previous week, ridge and histogram gradient boosting: each model's daily predictions schedule the same physical battery.
- Perfect foresight: actual prices are the decision vector. It is labeled as an infeasible-information upper bound, never as a deployable strategy.

Perfect foresight must have net modeled value at least as high as every other feasible schedule for the same day and configuration. The backtest asserts this property. Sensitivity uses perfect foresight to isolate efficiency, power/energy ratio and degradation assumptions from forecast error.

## Negative-price interpretation

Charging at a negative price correctly produces positive gross market revenue. Discharging at a negative price costs money. A battery can still rationally charge during a negative interval and later discharge—even at a modest positive price—because receiving payment while charging can offset losses. The binary mode forbids same-interval cycling but does not forbid legitimate across-time arbitrage.
