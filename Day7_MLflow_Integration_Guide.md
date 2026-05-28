# RetailPulse MLflow Integration Guide

This project uses MLflow to track model development locally across the three core modeling workstreams:

- Prophet revenue forecasting
- LSTM demand forecasting
- Customer clustering

## Local tracking setup

The helper functions in `retailpulse_mlflow_utils.py` create a local `mlruns/` folder at the project root and set the experiment name to `RetailPulse`.

## Logged experiments

The tracking script `retailpulse_mlflow_tracking.py` logs:

- Parameters for each model run
- Accuracy metrics such as MAPE, RMSE, and MAE
- Artifacts including forecast CSVs, reports, and cluster outputs
- Registered model versions for the Prophet, LSTM, and KMeans runs

## Model registry targets

The following model names are used in the local MLflow registry:

- `RetailPulseProphetForecast`
- `RetailPulseLSTMForecast`
- `RetailPulseCustomerKMeans`

## Operational notes

- Run the forecasting and clustering pipelines before the MLflow tracking script.
- The tracking summary is saved to `processed/mlflow_tracking_summary.md`.
- If the file-based registry backend is not available in another environment, the scripts still save the artifacts and report the registration attempt result.
