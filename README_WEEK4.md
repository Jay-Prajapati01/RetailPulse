# RetailPulse Week 4

This file summarizes the production and deployment layer added in Week 4.

## Local production stack

- `docker/Dockerfile.streamlit`
- `docker/Dockerfile.mlflow`
- `docker/Dockerfile.airflow`
- `docker/docker-compose.yml`
- `.env.template`
- `scripts/init_airflow.sh`
- `docs/Docker-setup.md`

## Kubernetes

- `k8s/namespace.yaml`
- `k8s/configmap.yaml`
- `k8s/secret.template.yaml`
- `k8s/streamlit-deployment.yaml`
- `k8s/mlflow-deployment.yaml`
- `k8s/postgres-deployment.yaml`
- `k8s/redis-deployment.yaml`
- `k8s/airflow-deployment.yaml`
- `k8s/ingress.yaml`

## CI/CD

- `.github/workflows/ci.yml`
- `.github/workflows/docker-build.yml`

## Monitoring and validation

- `monitoring/prometheus/prometheus.yml`
- `monitoring/grafana/provisioning/datasources/datasource.yml`
- `monitoring/grafana/provisioning/dashboards/dashboard.yml`
- `performance/locustfile.py`
- `docs/Streamlit-Cloud.md`
- `docs/Week4-Production.md`
