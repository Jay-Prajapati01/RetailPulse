# RetailPulse monitoring stack

This folder contains a free-tier compatible Prometheus + Grafana OSS observability setup.

## Run locally

```powershell
docker compose -f monitoring/docker-compose.monitoring.yml up -d
```

## Services

- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000
- cAdvisor: http://localhost:8081

## Notes

- Grafana uses the default `admin` / `admin` credentials in local development.
- The Prometheus config is intentionally minimal so it can run on a student laptop.
