# RetailPulse LSTM Forecasting Evaluation

## Test Metrics

|        MAPE |    RMSE |     MAE |
|------------:|--------:|--------:|
| 3.81841e+13 | 33788.3 | 25600.1 |

## Recent Predictions

| date       |   actual |   predicted |   absolute_error |
|:-----------|---------:|------------:|-----------------:|
| 2011-11-30 |  60127   |     25572.7 |        34554.2   |
| 2011-12-01 |  52197.3 |     25631   |        26566.3   |
| 2011-12-02 |  57664.1 |     25653.3 |        32010.8   |
| 2011-12-03 |      0   |     25677.1 |        25677.1   |
| 2011-12-04 |  24621.4 |     25545.1 |          923.647 |
| 2011-12-05 |  88742   |     25465.1 |        63276.9   |
| 2011-12-06 |  56713.2 |     25574.9 |        31138.3   |
| 2011-12-07 |  75439.2 |     25626   |        49813.1   |
| 2011-12-08 |  82495   |     25708.4 |        56786.6   |
| 2011-12-09 | 200939   |     25801   |       175138     |

## Business Interpretation

- The test split is strictly chronological, so the reported error reflects genuine forward-looking performance.
- The LSTM captures nonlinear sequence effects that a linear baseline may miss, but it still depends heavily on stable demand patterns.
- If the model underperforms Prophet, the business should prefer the simpler model for deployment until more exogenous signals are available.
- Re-training should happen on a rolling basis because retail demand shifts with promotions and seasonality.