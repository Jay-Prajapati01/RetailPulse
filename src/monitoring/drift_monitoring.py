from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

try:  # pragma: no cover - version-dependent import path
    from evidently import Report
    from evidently.presets.drift import DataDriftPreset
    from evidently.presets.regression import RegressionPreset
except Exception:  # pragma: no cover
    from evidently import Report
    from evidently.presets.drift import DataDriftPreset
    from evidently.presets.regression import RegressionPreset

from retailpulse_feature_pipeline import PROCESSED_DIR
from retailpulse_mlflow_utils import log_json_artifact, log_text_artifact, setup_mlflow, start_mlflow_run

ROOT_DIR = Path(__file__).resolve().parents[2]
MONITORING_DIR = ROOT_DIR / "monitoring"
DRIFT_REPORTS_DIR = MONITORING_DIR / "drift_reports"
FIGURES_DIR = PROCESSED_DIR / "figures"


@dataclass
class DriftMonitoringResult:
    data_drift_html: Path
    prediction_drift_html: Path
    monitoring_dashboard_html: Path
    summary_json: Path
    metrics: dict[str, float]


def ensure_monitoring_dirs() -> None:
    DRIFT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MONITORING_DIR.mkdir(parents=True, exist_ok=True)


def load_reference_current_daily_sales(split_ratio: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_sales = pd.read_csv(PROCESSED_DIR / "daily_sales_forecasting.csv", parse_dates=["date"])
    daily_sales = daily_sales.sort_values("date").reset_index(drop=True)
    split_index = max(int(len(daily_sales) * split_ratio), 1)
    reference = daily_sales.iloc[:split_index].copy()
    current = daily_sales.iloc[split_index:].copy()
    return reference, current


def load_reference_current_forecasts(split_ratio: float = 0.5) -> tuple[pd.DataFrame, pd.DataFrame]:
    forecast_frame = pd.read_csv(PROCESSED_DIR / "ensemble_evaluation_forecast.csv", parse_dates=["date"])
    forecast_frame = forecast_frame.sort_values("date").reset_index(drop=True)
    split_index = max(int(len(forecast_frame) * split_ratio), 1)
    reference = forecast_frame.iloc[:split_index].copy()
    current = forecast_frame.iloc[split_index:].copy()
    return reference, current


def build_evidently_report(reference_data: pd.DataFrame, current_data: pd.DataFrame, metrics: list[Any], output_path: Path) -> Path:
    report = Report(metrics=metrics)
    snapshot = report.run(reference_data=reference_data, current_data=current_data)
    snapshot.save_html(str(output_path))
    return output_path


def summarize_drift(reference_data: pd.DataFrame, current_data: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    summary = {}
    for column in columns:
        ref = pd.to_numeric(reference_data[column], errors="coerce")
        cur = pd.to_numeric(current_data[column], errors="coerce")
        ref_mean = float(ref.mean()) if ref.notna().any() else 0.0
        cur_mean = float(cur.mean()) if cur.notna().any() else 0.0
        ref_std = float(ref.std(ddof=0)) if ref.notna().any() else 0.0
        cur_std = float(cur.std(ddof=0)) if cur.notna().any() else 0.0
        mean_shift = abs(cur_mean - ref_mean) / max(abs(ref_mean), 1e-6)
        std_shift = abs(cur_std - ref_std) / max(abs(ref_std), 1e-6)
        summary[f"{column}_mean_shift"] = float(mean_shift)
        summary[f"{column}_std_shift"] = float(std_shift)
    return summary


def save_dashboard(
    reference_sales: pd.DataFrame,
    current_sales: pd.DataFrame,
    metrics: dict[str, float],
    output_path: Path,
    data_drift_html_name: str,
    prediction_drift_html_name: str,
) -> Path:
    image_path = output_path.with_suffix(".png")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    axes[0].plot(reference_sales["date"], reference_sales["total_price"], label="Reference", color="#1f77b4")
    axes[0].plot(current_sales["date"], current_sales["total_price"], label="Current", color="#d62728")
    axes[0].set_title("Revenue Drift")
    axes[0].legend()

    axes[1].bar(["Data Drift", "Prediction Drift"], [metrics["data_drift_score"], metrics["prediction_drift_score"]], color=["#ff7f0e", "#2ca02c"])
    axes[1].set_title("Drift Scores")

    axes[2].plot(reference_sales["date"], reference_sales["quantity"].rolling(7).mean(), label="Reference", color="#1f77b4")
    axes[2].plot(current_sales["date"], current_sales["quantity"].rolling(7).mean(), label="Current", color="#d62728")
    axes[2].set_title("Smoothed Quantity Trend")
    axes[2].legend()

    axes[3].axis("off")
    axes[3].text(
        0.02,
        0.9,
        "Business Interpretation\n\n"
        f"Model reliability score: {metrics['model_reliability_score']:.3f}\n"
        f"Forecasting degradation risk: {metrics['forecasting_degradation_risk']:.3f}\n"
        f"Customer behavior shift proxy: {metrics['behavior_shift_proxy']:.3f}",
        fontsize=12,
        va="top",
    )

    fig.suptitle("RetailPulse Monitoring Dashboard", fontsize=16)
    plt.tight_layout()
    fig.savefig(image_path, dpi=160)
    plt.close(fig)
    html_content = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>RetailPulse Monitoring Dashboard</title>
    <style>
        body {{ font-family: Segoe UI, Arial, sans-serif; margin: 0; padding: 24px; background: #0f172a; color: #e5e7eb; }}
        .card {{ background: #111827; border: 1px solid #334155; border-radius: 16px; padding: 20px; margin-bottom: 16px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }}
        .metric {{ background: rgba(255,255,255,0.04); border-radius: 12px; padding: 14px; }}
        a {{ color: #38bdf8; }}
        img {{ max-width: 100%; border-radius: 14px; border: 1px solid #334155; }}
    </style>
</head>
<body>
    <div class=\"card\">
        <h1>RetailPulse Monitoring Dashboard</h1>
        <p>Overview of data drift, prediction drift, and behavior shifts detected in the current monitoring window.</p>
    </div>
    <div class=\"card grid\">
        <div class=\"metric\"><strong>Model Reliability</strong><div>{metrics['model_reliability_score']:.3f}</div></div>
        <div class=\"metric\"><strong>Forecast Degradation Risk</strong><div>{metrics['forecasting_degradation_risk']:.3f}</div></div>
        <div class=\"metric\"><strong>Behavior Shift Proxy</strong><div>{metrics['behavior_shift_proxy']:.3f}</div></div>
        <div class=\"metric\"><strong>Data Drift Score</strong><div>{metrics['data_drift_score']:.3f}</div></div>
    </div>
    <div class=\"card\">
        <h2>Dashboard Image</h2>
        <img src=\"{image_path.name}\" alt=\"Monitoring dashboard chart\" />
    </div>
    <div class=\"card\">
        <h2>Reports</h2>
        <ul>
            <li><a href=\"{data_drift_html_name}\">Data drift report</a></li>
            <li><a href=\"{prediction_drift_html_name}\">Prediction drift report</a></li>
        </ul>
    </div>
</body>
</html>"""
    output_path.write_text(html_content, encoding="utf-8")
    return output_path


def run_drift_monitoring_pipeline() -> DriftMonitoringResult:
    ensure_monitoring_dirs()
    reference_sales, current_sales = load_reference_current_daily_sales()
    reference_forecasts, current_forecasts = load_reference_current_forecasts()

    data_drift_html = DRIFT_REPORTS_DIR / "data_drift_report.html"
    prediction_drift_html = DRIFT_REPORTS_DIR / "prediction_drift_report.html"
    monitoring_dashboard_html = DRIFT_REPORTS_DIR / "monitoring_dashboard.html"

    data_drift_columns = ["total_price", "quantity", "invoice_count", "customer_count", "day_of_week", "is_weekend", "month", "year", "week"]
    prediction_columns = ["total_price", "prophet_yhat", "lstm_yhat", "ensemble_yhat"]

    build_evidently_report(reference_sales[data_drift_columns], current_sales[data_drift_columns], [DataDriftPreset()], data_drift_html)
    prediction_reference = reference_forecasts[["total_price", "ensemble_yhat", "prophet_yhat", "lstm_yhat"]].rename(columns={"total_price": "target", "ensemble_yhat": "prediction"})
    prediction_current = current_forecasts[["total_price", "ensemble_yhat", "prophet_yhat", "lstm_yhat"]].rename(columns={"total_price": "target", "ensemble_yhat": "prediction"})
    build_evidently_report(prediction_reference, prediction_current, [DataDriftPreset()], prediction_drift_html)

    drift_summary = summarize_drift(reference_sales, current_sales, ["total_price", "quantity", "invoice_count", "customer_count"])
    data_drift_score = float(np.mean(list(drift_summary.values())))
    prediction_drift_score = float(abs(prediction_current["prediction"].mean() - prediction_reference["prediction"].mean()) / max(abs(prediction_reference["prediction"].mean()), 1e-6))
    metrics = {
        "data_drift_score": data_drift_score,
        "prediction_drift_score": prediction_drift_score,
        "model_reliability_score": max(0.0, 1.0 - min(data_drift_score, 1.0) - min(prediction_drift_score, 1.0)),
        "forecasting_degradation_risk": min(1.0, (data_drift_score + prediction_drift_score) / 2),
        "behavior_shift_proxy": float(np.mean([drift_summary.get("total_price_mean_shift", 0.0), drift_summary.get("customer_count_mean_shift", 0.0)])),
    }

    dashboard_path = save_dashboard(reference_sales, current_sales, metrics, monitoring_dashboard_html, data_drift_html.name, prediction_drift_html.name)
    summary_json = DRIFT_REPORTS_DIR / "drift_monitoring_summary.json"
    summary_json.write_text(json.dumps({"metrics": metrics, "drift_summary": drift_summary}, indent=2), encoding="utf-8")

    setup_mlflow("RetailPulse")
    with start_mlflow_run("day12_drift_monitoring") as run:
        mlflow.log_metrics(metrics)
        log_json_artifact({"metrics": metrics, "drift_summary": drift_summary}, "drift_monitoring_summary.json")
        log_text_artifact((DRIFT_REPORTS_DIR / "monitoring_dashboard.html").read_text(encoding="utf-8"), "monitoring_dashboard.html")

    return DriftMonitoringResult(
        data_drift_html=data_drift_html,
        prediction_drift_html=prediction_drift_html,
        monitoring_dashboard_html=dashboard_path,
        summary_json=summary_json,
        metrics=metrics,
    )
