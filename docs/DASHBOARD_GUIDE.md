# RetailPulse Dashboard Guide

The Streamlit dashboard is the product-experience layer of RetailPulse.

## Pages

- Home
- Forecasting
- Customer Segmentation
- Churn Analytics
- Inventory Optimization
- Monitoring
- Reports
- Settings

## Design Rule

This is an analytics SaaS interface, not a notebook dump. Keep the UI focused on business signals, not raw model internals.

## Entry Point

Run the dashboard with:

```powershell
streamlit run retailpulse_dashboard.py
```

## Data Sources

- `processed/` contains the generated analytics artifacts.
- `monitoring/drift_reports/` contains Evidently reports.
- `graphify-out/` remains the repository intelligence layer.
