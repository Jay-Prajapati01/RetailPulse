# Developer Setup

## Environment

- Python 3.12+
- `.venv` virtual environment
- Graphify CLI installed with `uv tool install graphifyy`

## Core Setup

```powershell
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
graphify update .
```

## Validation Order

1. Run the preprocessing pipeline.
2. Run Prophet and LSTM forecasting.
3. Run the hybrid ensemble.
4. Run churn prediction.
5. Run inventory optimization.
6. Run Optuna tuning.
7. Run drift monitoring.
8. Refresh Graphify.
