# RetailPulse Architecture

RetailPulse is organized as a layered AI platform rather than a notebook collection.

## Layers

1. Raw retail transactions are cleaned and standardized in `retailpulse_feature_pipeline.py`.
2. Forecasting modules consume daily and weekly series and produce Prophet, LSTM, and ensemble forecasts.
3. The churn pipeline converts customer RFM and activity data into a binary churn classifier.
4. The inventory optimizer consumes future demand forecasts and converts them into stock actions.
5. The Optuna module tunes model and ensemble hyperparameters.
6. The monitoring module uses Evidently to compare reference and current windows.
7. Airflow DAGs orchestrate the runnable modules.
8. Graphify provides the persistent repository intelligence layer.

## Design Rules

- Keep shared preprocessing centralized.
- Keep training logic in reusable Python modules.
- Keep notebook files as thin orchestration layers.
- Log every major experiment into MLflow.
- Refresh Graphify after structural changes.
