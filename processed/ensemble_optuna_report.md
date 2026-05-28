# Day 11 - Optuna Ensemble Weight Optimization

|   prophet |    lstm |
|----------:|--------:|
|   0.13176 | 0.86824 |

| metric        |           value |
|:--------------|----------------:|
| ensemble_rmse | 45895.3         |
| ensemble_mae  | 34615.8         |
| ensemble_mape |     3.46445e+13 |

The best ensemble is selected by minimizing validation RMSE on the forecast comparison window.