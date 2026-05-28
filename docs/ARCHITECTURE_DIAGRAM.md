# RetailPulse Architecture Diagram

## System Overview

```mermaid
graph TD
    subgraph User Layer
      U[User Browser]
    end

    subgraph App Layer
      S[Streamlit Dashboard]
      MLF[MLflow Tracking]
      AFW[Airflow Webserver]
      AFS[Airflow Scheduler]
    end

    subgraph Data Layer
      PG[(PostgreSQL)]
      RD[(Redis)]
      ART[(Processed Artifacts)]
      MLR[(MLruns / Models)]
    end

    subgraph Ops Layer
      GH[GitHub Actions]
      GHCR[GitHub Container Registry]
      PROM[Prometheus]
      GRAF[Grafana]
      LOC[Locust]
    end

    U --> S
    S --> ART
    S --> MLF
    AFW --> PG
    AFS --> PG
    AFS --> ART
    AFS --> MLR
    MLF --> MLR

    GH --> GHCR
    GH --> S
    GH --> AFS

    PROM --> S
    PROM --> MLF
    PROM --> AFW
    GRAF --> PROM
    LOC --> S
```

## Deployment Targets

- Local: Docker Compose (full stack)
- Local cluster: Minikube or kind (Kubernetes manifests)
- Public app: Streamlit Cloud (dashboard only)
