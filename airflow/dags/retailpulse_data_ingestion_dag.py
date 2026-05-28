from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

from _shared import DEFAULT_ARGS


def run_data_ingestion():
    from retailpulse_feature_pipeline import run_and_save_pipeline

    return run_and_save_pipeline()


with DAG(
    dag_id="retailpulse_data_ingestion",
    default_args=DEFAULT_ARGS,
    description="RetailPulse raw data ingestion and preprocessing entry DAG",
    schedule_interval="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["retailpulse", "ingestion"],
) as dag:
    ingest = PythonOperator(task_id="run_data_ingestion", python_callable=run_data_ingestion)
