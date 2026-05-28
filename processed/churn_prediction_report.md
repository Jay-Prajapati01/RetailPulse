# RetailPulse Churn Prediction Report

## Churn Logic

Customers are labeled as churned when inactivity reaches at least 90 days.

## Evaluation Metrics

|   roc_auc |   precision |   recall |   f1_score |
|----------:|------------:|---------:|-----------:|
|   0.99998 |    0.996005 | 0.997333 |   0.996669 |

## Top Risk Customers

|   recency_days |   frequency |   total_revenue |   churn_probability | risk_band   |
|---------------:|------------:|----------------:|--------------------:|:------------|
|            436 |           3 |          979.8  |            0.999847 | High        |
|            367 |           3 |          988.44 |            0.999844 | High        |
|            360 |           3 |         1161.9  |            0.999843 | High        |
|            355 |           5 |         1580.54 |            0.999842 | High        |
|            335 |           3 |         1017.33 |            0.999838 | High        |
|            428 |           4 |         1199.31 |            0.999835 | High        |
|            503 |           2 |          736.93 |            0.999835 | High        |
|            184 |           4 |         1235.62 |            0.999831 | High        |
|            394 |           1 |          381.09 |            0.999831 | High        |
|            283 |           2 |          685.9  |            0.999831 | High        |

## SHAP Feature Ranking

| feature                |   mean_abs_shap |
|:-----------------------|----------------:|
| recency_days           |       6.56203   |
| inactivity_duration    |       0.961198  |
| purchase_velocity      |       0.330815  |
| monetary_per_day       |       0.216086  |
| basket_value_stability |       0.193262  |
| revenue_per_order      |       0.170615  |
| units_per_order        |       0.125956  |
| avg_purchase_interval  |       0.109629  |
| total_revenue          |       0.064094  |
| orders_per_day         |       0.0529304 |
| frequency              |       0.0485432 |
| avg_unit_price         |       0.0424608 |

## Business Interpretation

- High-risk customers are concentrated where recency is high and purchase frequency is low.
- These customers should be prioritized for retention campaigns, win-back offers, and service outreach.
- Recency-driven churn is the strongest signal in this dataset, so operational contact timing matters.
- The SHAP ranking can be used to explain individual retention decisions to the business team.