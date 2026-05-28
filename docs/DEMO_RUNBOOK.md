# RetailPulse Demo Runbook

## Demo objective

Show RetailPulse as a production-style, free-tier deployable AI platform.

## Pre-demo checklist

1. Ensure `.venv` is active.
2. Ensure required artifacts exist in `processed/`.
3. Ensure dashboard launches locally.
4. Keep architecture and Week 4 docs open.

## Demo flow (15-20 minutes)

1. Problem framing (1 minute)
- Explain retail forecasting, churn, and inventory optimization challenge.

2. Platform architecture (2 minutes)
- Show `docs/ARCHITECTURE_DIAGRAM.md`.
- Explain app layer, data layer, and operations layer.

3. Dashboard walkthrough (6-8 minutes)
- Home and Forecasting
- Segmentation and Churn
- Inventory and Monitoring
- Reports export flow

4. MLOps and automation (4-5 minutes)
- Show MLflow tracking summary.
- Show Airflow orchestration assets.
- Show GitHub Actions workflows.

5. Production readiness (2-3 minutes)
- Show Docker, Kubernetes, monitoring, and load-testing assets.
- Explain free-tier deployment path via Streamlit Cloud.

## Demo commands

```powershell
# local dashboard
streamlit run retailpulse_dashboard.py

# Docker stack (when Docker Desktop is available)
scripts/start_week4.ps1 -Build

# Kubernetes (when Minikube/kind is available)
scripts/k8s_minikube_start.ps1
```

## FAQ answers to prepare

- Why Streamlit Cloud for public deployment?
- Why Docker and Kubernetes both exist in the repo?
- How drift and model reliability are monitored?
- How CI/CD ensures deployment quality?
