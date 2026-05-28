# Day 11 - Optuna Churn Optimization

|   n_estimators |   max_depth |   learning_rate |   subsample |   colsample_bytree |   min_child_weight |   reg_lambda |   reg_alpha |
|---------------:|------------:|----------------:|------------:|-------------------:|-------------------:|-------------:|------------:|
|            385 |           5 |        0.129287 |    0.988739 |           0.896667 |            1.67022 |      3.44229 |    0.153593 |

| metric             |    value |
|:-------------------|---------:|
| best_roc_auc       | 0.999981 |
| validation_roc_auc | 0.999981 |

The best XGBoost configuration is the one with the highest validation ROC AUC.