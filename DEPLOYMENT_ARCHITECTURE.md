# RetailPulse Deployment Architecture

## Overview

RetailPulse has been converted to a **visualization-only dashboard** optimized for stable deployment on Streamlit Cloud. This architecture is ideal for portfolio/demo purposes.

## Architecture Decision

### Previous Architecture (Runtime Generation)
```
Dashboard Startup
    ↓
Run Preprocessing Pipeline
    ↓
Run Forecasting Pipeline
    ↓
Run Inventory Optimization
    ↓
Run Churn Prediction
    ↓
Generate Artifacts Dynamically
    ↓
Display Analytics
```

**Issues:**
- Complex orchestration dependencies
- Runtime failures on Streamlit Cloud
- Heavy ML dependencies (torch, mlflow, evidently, prophet)
- Slow startup times
- Deployment instability

### Current Architecture (Pre-Generated Artifacts)
```
Local Development
    ↓
Generate All Analytics Once
    ↓
Commit Artifacts to Git
    ↓
Push to GitHub
    ↓
Streamlit Cloud Deployment
    ↓
Load Existing Files Only
    ↓
Display Analytics Instantly
```

**Benefits:**
- ✅ Stable and reliable deployment
- ✅ Fast dashboard startup
- ✅ No runtime pipeline failures
- ✅ Lightweight dependencies
- ✅ Portfolio-ready presentation

## What Changed

### 1. Gitignore Updates
**File:** `.gitignore`

**Changes:**
- ✅ Keep `processed/` directory (except large CSVs)
- ✅ Keep `models/` directory
- ✅ Keep `monitoring/drift_reports/` directory
- ❌ Exclude `processed/retail_transactions_raw.csv` (104MB)
- ❌ Exclude `processed/retail_transactions_cleaned.csv` (152MB)

### 2. Dashboard Pages Modified
All dashboard pages now **load existing artifacts only** (no runtime generation):

- `src/dashboard/pages/forecasting.py` - Removed `ensure_artifacts("forecasting")`
- `src/dashboard/pages/churn.py` - Removed `ensure_artifacts("churn")`
- `src/dashboard/pages/inventory.py` - Removed `ensure_artifacts("inventory")`
- `src/dashboard/pages/segmentation.py` - Removed `ensure_artifacts("segmentation")`
- `src/dashboard/pages/monitoring.py` - Removed `ensure_artifacts("monitoring")`
- `src/dashboard/pages/reports.py` - Removed `ensure_artifacts(...)`

### 3. Artifact Bootstrap Disabled
**File:** `src/dashboard/artifact_bootstrap.py`

Added documentation noting that this module is **disabled for deployment** but preserved for local development.

### 4. Dashboard Entry Point
**File:** `retailpulse_dashboard.py`

Updated documentation to reflect visualization-only architecture.

## Committed Artifacts

### Analytics Outputs (processed/)
- ✅ Forecasting CSVs (ensemble, LSTM, Prophet predictions)
- ✅ Churn prediction CSVs and metrics
- ✅ Inventory recommendations and alerts
- ✅ Customer segmentation labels and personas
- ✅ RFM analysis outputs
- ✅ Drift monitoring summaries

### Reports (processed/)
- ✅ Markdown reports for all pipelines
- ✅ Optuna hyperparameter tuning reports
- ✅ Model evaluation reports

### Visualizations (processed/figures/)
- ✅ SHAP waterfall and summary plots
- ✅ Confusion matrices
- ✅ Forecast comparison charts
- ✅ Time series decomposition plots
- ✅ Cluster visualization (PCA, t-SNE)
- ✅ Inventory health dashboards
- ✅ Optuna optimization history

### Models (processed/models/)
- ✅ Trained XGBoost churn model
- ✅ LSTM forecasting model
- ✅ Prophet model
- ✅ Customer segmentation models (KMeans, DBSCAN)
- ✅ Scalers and encoders

### Monitoring (monitoring/drift_reports/)
- ✅ Data drift HTML reports
- ✅ Prediction drift HTML reports
- ✅ Monitoring dashboard HTML

## Deployment Requirements

### Lightweight Dependencies
**File:** `requirements.txt`

Only visualization and inference packages:
```
streamlit==1.35.0
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.1
plotly==5.22.0
matplotlib==3.9.0
seaborn==0.13.2
scipy==1.13.1
joblib==1.4.2
statsmodels==0.14.2
xgboost==2.1.0
lightgbm==4.5.0
pyyaml==6.0.2
openpyxl==3.1.5
reportlab==4.2.2
altair==5.3.0
shap==0.45.1
tabulate==0.9.0
```

**Removed from deployment:**
- ❌ torch / pytorch-lightning (training only)
- ❌ mlflow (experiment tracking only)
- ❌ evidently (drift generation only)
- ❌ prophet (training only)
- ❌ optuna (hyperparameter tuning only)

### Python Version
**File:** `.python-version`
```
3.11
```

### System Dependencies
**File:** `packages.txt`
```
libgomp1
```

## Local Development Workflow

### Regenerating Artifacts
If you need to update analytics locally:

```bash
# 1. Run all pipelines locally
python retailpulse_feature_pipeline.py
python retailpulse_forecasting_pipeline.py
python retailpulse_inventory_optimization.py
python retailpulse_customer_segmentation.py
python src/churn/churn_pipeline.py

# 2. Verify outputs
ls -lh processed/
ls -lh monitoring/drift_reports/

# 3. Commit updated artifacts
git add processed/ monitoring/
git commit -m "Updated pre-generated artifacts"
git push origin main
```

### Testing Dashboard Locally
```bash
# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run retailpulse_dashboard.py
```

## Streamlit Cloud Deployment

### Deployment Settings
- **Python version:** 3.11
- **Main file:** `retailpulse_dashboard.py`
- **Requirements:** `requirements.txt`
- **System packages:** `packages.txt`

### Expected Behavior
1. Dashboard loads instantly (no pipeline execution)
2. All pages display pre-generated analytics
3. No runtime errors from missing dependencies
4. Fast page navigation
5. Stable performance

## Future Enhancements

### Option 1: Keep Current Architecture
Best for portfolio/demo purposes. No changes needed.

### Option 2: Add Runtime Orchestration
If you need dynamic artifact generation in production:

1. Re-enable `ensure_artifacts()` calls in pages
2. Add heavy dependencies back to requirements
3. Implement proper error handling and caching
4. Consider using Streamlit Cloud's paid tier for more resources
5. Or deploy to AWS/GCP with proper infrastructure

### Option 3: Hybrid Approach
- Keep visualization-only for Streamlit Cloud (public demo)
- Deploy full orchestration version separately (internal use)
- Use CI/CD to regenerate artifacts periodically

## Summary

RetailPulse is now a **stable, fast, portfolio-ready analytics dashboard** that:
- Loads pre-generated ML artifacts from Git
- Displays professional analytics instantly
- Deploys reliably on Streamlit Cloud
- Requires minimal dependencies
- Provides excellent user experience

This architecture is **ideal for showcasing your ML capabilities** without the complexity of production orchestration.
