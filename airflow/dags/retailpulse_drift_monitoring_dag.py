from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

from _shared import DEFAULT_ARGS


def run_drift_monitoring():
    from retailpulse_drift_monitoring import run_drift_monitoring_pipeline

    return run_drift_monitoring_pipeline()


with DAG(
    dag_id="retailpulse_drift_monitoring",
    default_args=DEFAULT_ARGS,
    description="RetailPulse drift monitoring DAG",
    schedule_interval="0 5 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["retailpulse", "monitoring"],
) as dag:
    monitor = PythonOperator(task_id="run_drift_monitoring", python_callable=run_drift_monitoring)
