# Week 4 production plan

This document captures the final production transformation for RetailPulse.

## Day 22

Dockerize the local stack.

- Streamlit dashboard
- MLflow tracking server
- PostgreSQL
- Redis
- Airflow webserver and scheduler

## Day 23

Add Kubernetes manifests for Minikube or kind.

- Namespaces
- Deployments
- Services
- Ingress
- ConfigMap and Secret templates

## Day 24

Add CI/CD automation with GitHub Actions.

- Repository validation
- Python checks
- YAML validation
- Docker build validation

## Day 25

Prepare Streamlit Cloud deployment.

- Deployment-safe requirements
- Dashboard entrypoint verification
- Fallback behavior for missing artifacts

## Day 26

Add monitoring stack scaffolding.


## Streamlit runtime validation

This repository includes a lightweight Streamlit runtime CI job at `.github/workflows/streamlit-runtime.yml` which:

- Installs `requirements-streamlit.txt` (lightweight runtime)
- Starts the Streamlit app and performs a health-check against `/_stcore/health`
- Uploads Streamlit logs for debugging

Use this workflow to ensure the dashboard will start in free-tier environments before running heavier Docker builds.

## Day 27

Add performance validation.
- Stability review

## Day 28

Final polish.

- Documentation
- Architecture overview
- Demo-readiness checklist
