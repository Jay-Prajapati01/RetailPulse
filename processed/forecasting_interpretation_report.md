# RetailPulse Forecasting Readiness Report

## Stationarity Results

|   series_length |   adf_statistic |      p_value |   lags_used |   observations |   critical_1% |   critical_5% |   critical_10% | stationary   |
|----------------:|----------------:|-------------:|------------:|---------------:|--------------:|--------------:|---------------:|:-------------|
|             739 |        -2.00678 |   0.283608   |          20 |            718 |      -3.43949 |      -2.86557 |       -2.56892 | False        |
|             106 |        -3.67497 |   0.00447929 |           0 |            105 |      -3.49422 |      -2.88949 |       -2.58168 | True         |
|             739 |       nan       | nan          |           0 |            738 |      -3.43924 |      -2.86546 |       -2.56886 | False        |

## Interpretation

- Daily and weekly revenue series should be inspected for trend and seasonality before model selection.
- A p-value below 0.05 indicates stationarity and supports direct ARIMA-style modeling with fewer differencing steps.
- Non-stationary series should be differenced or transformed before forecasting.
- Product-level demand is highly sparse, so the highest-revenue products are the best candidates for an initial forecast benchmark.