#!/usr/bin/env bash
set -euo pipefail

echo "Initializing Airflow DB and creating admin user (inside airflow-webserver container)..."

# This script is intended to be run from the host with docker-compose up -d already started,
# or run inside the airflow-webserver container via: docker compose run --rm airflow-webserver bash /opt/airflow/scripts/init_airflow.sh

airflow db upgrade
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin

echo "Airflow initialized."
