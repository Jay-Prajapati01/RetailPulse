# Docker setup for RetailPulse

This document explains how to run the local Dockerized RetailPulse stack. The stack includes:

- Streamlit dashboard (Streamlit)
- MLflow tracking server (MLflow)
- PostgreSQL (for Airflow and optionally MLflow)
- Airflow webserver + scheduler

Prerequisites
-------------

1. Install Docker Desktop (Windows) and enable WSL2.
2. Install Docker Compose (Docker Desktop includes it).
3. Clone this repository and open a terminal in the project root.
4. Copy the environment template:

```powershell
copy .env.template .env
# Edit .env as needed (change passwords and keys)
```

Bring up the stack
------------------

From the project root run:

```powershell
docker compose -f docker/docker-compose.yml up --build -d
```

Notes
-----
- The first run will build three images (streamlit, mlflow, airflow). It may take several minutes.
- Airflow uses `LocalExecutor` in this compose; this is simpler for local development.
- MLflow uses a persistent SQLite backend in `/mlflow` for simplicity and free-tier compatibility.

Initialize services
-------------------

Run the one-shot Airflow init service the first time, or after clearing volumes:

```powershell
docker compose -f docker/docker-compose.yml run --rm airflow-init
```

Then start the long-running services:

```powershell
docker compose -f docker/docker-compose.yml up -d
```

Verify services
---------------

- Streamlit: http://localhost:8501
- MLflow: http://localhost:5000
- Airflow: http://localhost:8080 (user: admin / password: admin)

Stopping and cleaning
---------------------

```powershell
docker compose -f docker/docker-compose.yml down -v
```

Troubleshooting
---------------
- If ports are in use, change them in `docker-compose.yml` or stop conflicting services.
- If the Airflow container fails, inspect logs: `docker compose -f docker/docker-compose.yml logs airflow-webserver`.
- If Docker Compose complains about missing Docker CLI on Windows, ensure Docker Desktop is installed and added to PATH.
