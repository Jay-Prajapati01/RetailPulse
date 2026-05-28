from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

from _shared import DEFAULT_ARGS


def generate_inventory_report():
    from retailpulse_inventory_optimization import run_inventory_optimization_pipeline

    return run_inventory_optimization_pipeline()


with DAG(
    dag_id="retailpulse_inventory_report",
    default_args=DEFAULT_ARGS,
    description="RetailPulse inventory report generation DAG",
    schedule_interval="0 6 * * 1",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["retailpulse", "inventory"],
) as dag:
    report = PythonOperator(task_id="generate_inventory_report", python_callable=generate_inventory_report)
