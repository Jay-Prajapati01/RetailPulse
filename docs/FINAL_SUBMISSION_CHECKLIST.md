# Final Submission Checklist — RetailPulse Week 4

This file lists precise commands and verification steps to validate the Week 4 production readiness locally and in CI.

Prerequisites
- Docker Desktop (Windows) with WSL2 enabled
- Python 3.10 virtual environment
- kubectl + minikube/kind (optional for Kubernetes validation)

1) Start local Docker stack (build + run)

```powershell
# from repository root
.\scripts\start_week4.ps1 -Build
# or
docker compose -f docker/docker-compose.yml up --build -d
```

Verify services are healthy:

```powershell
docker compose -f docker/docker-compose.yml ps
docker compose -f docker/docker-compose.yml logs -f streamlit
```

2) Initialize Airflow (inside containers)

```powershell
docker compose -f docker/docker-compose.yml run --rm airflow-webserver bash /opt/airflow/scripts/init_airflow.sh
```

3) Start monitoring stack

```powershell
docker compose -f monitoring/docker-compose.monitoring.yml up -d
```

4) Smoke-check Streamlit

```powershell
curl --fail --retry 10 --retry-delay 2 http://localhost:8501/_stcore/health
```

5) Run lightweight Streamlit CI locally (optional)

```bash
# Validate Streamlit runtime using the lightweight requirements
pip install -r requirements-streamlit.txt
streamlit run retailpulse_dashboard.py
```

6) Kubernetes validation (optional)

```powershell
minikube start
kubectl apply -k k8s
kubectl get pods -n retailpulse -w
```

7) Regenerate Graphify outputs (requires `graphify` installed)

```powershell
.\scripts\update-graphify.ps1
# or
graphify update .
```

8) Common CI workflows
- `.github/workflows/validation.yml` — YAML & notebook validation
- `.github/workflows/streamlit-runtime.yml` — Streamlit start smoke test (lightweight deps)
- `.github/workflows/docker-build.yml` — Docker image build & push (GHCR requires secrets)

Acceptance criteria
- Streamlit UI loads and pages navigate without unhandled exceptions
- Monitoring health shows no critical alerts (or alerts are actionable)
- MLflow accessible on port 5000
- Airflow UI accessible on port 8080
- Docker images build successfully in CI (when secrets are configured)

If something fails, collect service logs and re-run the corresponding health checks above.
