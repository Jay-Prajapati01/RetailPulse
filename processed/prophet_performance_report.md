# RetailPulse Prophet Forecasting Report

## Best Configuration

| yearly_seasonality   | weekly_seasonality   | seasonality_mode   |   seasonality_prior_scale |   changepoint_prior_scale |
|:---------------------|:---------------------|:-------------------|--------------------------:|--------------------------:|
| True                 | True                 | additive           |                        10 |                      0.05 |

## Validation Metrics

|        MAPE |    RMSE |     MAE |
|------------:|--------:|--------:|
| 4.01079e+13 | 33095.3 | 20577.9 |

## Grid Search Summary

| yearly_seasonality   | weekly_seasonality   | seasonality_mode   |   seasonality_prior_scale |   changepoint_prior_scale |        MAPE |    RMSE |     MAE |
|:---------------------|:---------------------|:-------------------|--------------------------:|--------------------------:|------------:|--------:|--------:|
| True                 | True                 | additive           |                        10 |                      0.05 | 4.01079e+13 | 33095.3 | 20577.9 |
| True                 | True                 | multiplicative     |                        10 |                      0.05 | 4.06453e+13 | 33434.6 | 22721.9 |
| True                 | False                | additive           |                         5 |                      0.1  | 7.66938e+13 | 39223.8 | 26332   |
| True                 | False                | multiplicative     |                         5 |                      0.1  | 7.99351e+13 | 39461.9 | 27045.5 |

## Next 30 Days Forecast

| ds         |     yhat |   yhat_lower |   yhat_upper |
|:-----------|---------:|-------------:|-------------:|
| 2011-12-10 | 28645.2  |      459.895 |      57065.9 |
| 2011-12-11 | 43935.3  |    17680.9   |      71832.8 |
| 2011-12-12 | 58580.7  |    29375.1   |      88596.9 |
| 2011-12-13 | 60379.5  |    32561.5   |      88359.4 |
| 2011-12-14 | 51945    |    23508.2   |      80781.3 |
| 2011-12-15 | 55505.7  |    28335.3   |      83502   |
| 2011-12-16 | 43533.1  |    14293.5   |      72105.9 |
| 2011-12-17 |  8762.96 |   -19646.7   |      40648.3 |
| 2011-12-18 | 22944.5  |    -7332.11  |      53061.8 |
| 2011-12-19 | 36778.7  |     8882.78  |      65197.3 |

## Business Interpretation

- The best Prophet configuration is selected on the last 30 days of observed sales, which gives an honest out-of-sample test.
- Confidence intervals should be used as planning bands rather than exact revenue promises.
- If weekly seasonality dominates, RetailPulse should align inventory and campaign cadence to day-of-week effects.
- Yearly seasonality becomes more important as the business crosses holiday and seasonal buying cycles.