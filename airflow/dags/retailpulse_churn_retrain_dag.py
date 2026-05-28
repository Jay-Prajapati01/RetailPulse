from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

from _shared import DEFAULT_ARGS


def retrain_churn_model():
    from retailpulse_churn_prediction import run_churn_prediction_pipeline

    return run_churn_prediction_pipeline()


with DAG(
    dag_id="retailpulse_churn_retrain",
    default_args=DEFAULT_ARGS,
    description="RetailPulse churn retraining DAG",
    schedule_interval="0 3 * * 1",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["retailpulse", "churn"],
) as dag:
    retrain = PythonOperator(task_id="retrain_churn_model", python_callable=retrain_churn_model)
