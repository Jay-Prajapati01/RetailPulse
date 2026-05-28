# RetailPulse Airflow Orchestration

This folder contains the production-style workflow definitions for RetailPulse.

## Structure

- `dags/` contains modular DAG definitions.
- `logs/` is the default runtime log location.
- `plugins/` is reserved for custom operators, hooks, or macros.

## DAG Coverage

- data ingestion
- preprocessing
- forecasting retraining
- churn retraining
- drift monitoring
- inventory report generation

## Execution Model

The DAGs are designed to call the reusable Python modules under `src/`.
They include retries, scheduling metadata, and explicit task dependencies so the platform can be deployed into a real Airflow environment later without changing the core business logic.
