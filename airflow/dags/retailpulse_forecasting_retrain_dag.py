from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

from _shared import DEFAULT_ARGS


def retrain_forecasting_stack():
    from retailpulse_forecasting_pipeline import run_forecasting_pipeline
    from retailpulse_hybrid_forecasting_ensemble import run_hybrid_forecasting_ensemble

    run_forecasting_pipeline()
    return run_hybrid_forecasting_ensemble()


with DAG(
    dag_id="retailpulse_forecasting_retrain",
    default_args=DEFAULT_ARGS,
    description="RetailPulse forecasting retraining DAG",
    schedule_interval="0 2 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["retailpulse", "forecasting"],
) as dag:
    retrain = PythonOperator(task_id="retrain_forecasting_stack", python_callable=retrain_forecasting_stack)
