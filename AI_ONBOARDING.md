# AI Onboarding for RetailPulse

This repository uses Graphify as a persistent AI-readable intelligence layer. It is not a product feature of RetailPulse. Its purpose is to keep repository understanding stable across assistants, sessions, and platforms.

## Read First

When onboarding to this repo, read these files in order:

1. `graphify-out/GRAPH_REPORT.md`
2. `graphify-out/graph.json`
3. `graphify-out/graph.html`

If those files are stale or missing, regenerate them before making implementation changes.

## What Graphify Is Used For

Graphify captures the repository structure as an AI memory layer:

- architecture and subsystem boundaries
- module relationships and imports
- workflow and pipeline links
- dependency structure
- completed versus pending systems
- cross-file navigation for future assistants

## Why It Exists

RetailPulse is large enough that repeated manual explanation wastes time and increases inconsistency. Graphify reduces context loss by creating a shared reference point that future AI assistants can read immediately.

## Key Outputs

- `graphify-out/GRAPH_REPORT.md` is the human-readable summary and navigation layer.
- `graphify-out/graph.json` is the machine-readable repository memory source of truth.
- `graphify-out/graph.html` is the visual exploration view for architecture and dependency analysis.

## Current Week 2 Modules

- `src/forecasting/ensemble.py` and `retailpulse_hybrid_forecasting_ensemble.py`
- `src/churn/churn_pipeline.py` and `retailpulse_churn_prediction.py`
- `src/inventory/inventory_optimization.py` and `retailpulse_inventory_optimization.py`
- `RetailPulse_Day8_Hybrid_Ensemble_Forecasting.ipynb`
- `RetailPulse_Day9_Churn_Prediction.ipynb`
- `RetailPulse_Day10_Inventory_Optimization.ipynb`

## Current Week 3 Modules

- `src/mlops/optuna_optimization.py` and `retailpulse_optuna_optimization.py`
- `src/monitoring/drift_monitoring.py` and `retailpulse_drift_monitoring.py`
- `airflow/dags/retailpulse_data_ingestion_dag.py`
- `airflow/dags/retailpulse_forecasting_retrain_dag.py`
- `airflow/dags/retailpulse_churn_retrain_dag.py`
- `airflow/dags/retailpulse_drift_monitoring_dag.py`
- `airflow/dags/retailpulse_inventory_report_dag.py`
- `RetailPulse_Day11_Optuna_Optimization.ipynb`
- `RetailPulse_Day12_Drift_Monitoring.ipynb`
- `RetailPulse_Day13_Airflow_Orchestration.ipynb`

## How to Regenerate

From the repository root:

```powershell
graphify update .
```

To keep the graph continuously refreshed while files change:

```powershell
graphify watch .
```

VS Code tasks are also available under `.vscode/tasks.json`.

## Assistant Integration

This repo has VS Code Copilot Chat integration configured through Graphify. Future assistants should still start with the Graphify outputs above, then inspect source files only when they need implementation-level detail.

For other assistants, the recommended workflow is:

1. Read `graphify-out/GRAPH_REPORT.md`.
2. Inspect `graphify-out/graph.json` for exact relationships.
3. Use `graphify-out/graph.html` to explore architecture visually.
4. Open the relevant source files only after the graph points to them.

## Best Practices

- Regenerate Graphify outputs after major changes to forecasting, segmentation, churn, dashboards, orchestration, deployment, or monitoring.
- Treat `graphify-out/graph.json` as the memory layer and `GRAPH_REPORT.md` as the human summary.
- Keep the graph current whenever the repo structure changes materially.
- Do not use Graphify as a substitute for source code review when implementation details matter.

## Related Automation

- `scripts/update-graphify.ps1` rebuilds the graph on demand.
- `scripts/watch-graphify.ps1` keeps the graph updated during active editing.
- `.github/copilot-instructions.md` contains the Graphify instruction block for Copilot Chat.
- Git hook automation is optional and only works after the workspace is initialized as a git repository.
