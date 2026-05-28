from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "processed"
MODELS_DIR = PROCESSED_DIR / "models"
FIGURES_DIR = PROCESSED_DIR / "figures"


@dataclass
class InventoryRunResult:
    recommendations_path: Path
    alerts_path: Path
    metrics_path: Path
    report_path: Path
    stock_risk_chart_path: Path
    reorder_schedule_chart_path: Path
    dashboard_chart_path: Path


def load_forecasts() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensemble = pd.read_csv(PROCESSED_DIR / "ensemble_future_30d_forecast.csv", parse_dates=["date"])
    prophet = pd.read_csv(PROCESSED_DIR / "prophet_future_30d_forecast.csv", parse_dates=["ds"])
    lstm = pd.read_csv(PROCESSED_DIR / "lstm_future_30d_forecast.csv", parse_dates=["date"])
    return ensemble, prophet, lstm


def load_product_demand() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "product_daily_demand.csv", parse_dates=["date"])


def build_inventory_recommendations(lead_time_days: int = 7, service_level: float = 0.95, top_n: int = 10) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensemble, prophet, lstm = load_forecasts()
    product_demand = load_product_demand()

    z_score = 1.645 if service_level >= 0.95 else 1.28
    average_unit_price = float(np.nanmedian((ensemble["ensemble_yhat"] / ensemble["ensemble_quantity_forecast"].replace(0, np.nan)).to_numpy(dtype=float)))
    aggregate_units_forecast = float(ensemble[["prophet_quantity_forecast", "lstm_quantity_forecast", "ensemble_quantity_forecast"]].mean(axis=1).sum())
    forecast_days = max(len(ensemble), 1)

    product_summary = (
        product_demand.groupby(["StockCode", "Description"], as_index=False)
        .agg(
            total_units=("quantity", "sum"),
            total_revenue=("total_price", "sum"),
            mean_daily_units=("quantity", "mean"),
            std_daily_units=("quantity", "std"),
        )
        .sort_values("total_revenue", ascending=False)
        .head(top_n)
        .fillna(0)
        .reset_index(drop=True)
    )

    total_units = max(float(product_summary["total_units"].sum()), 1e-6)
    rows: list[dict[str, float | str]] = []
    for rank, row in product_summary.iterrows():
        demand_share = float(row["total_units"] / total_units)
        forecast_units = aggregate_units_forecast * demand_share
        avg_daily_units = float(max(row["mean_daily_units"], 1e-6))
        demand_std = float(row["std_daily_units"] if pd.notna(row["std_daily_units"]) else avg_daily_units * 0.35)
        safety_stock = z_score * demand_std * np.sqrt(lead_time_days)
        reorder_point = avg_daily_units * lead_time_days + safety_stock
        simulated_current_stock = max(int(round(avg_daily_units * lead_time_days * (0.7 + 0.05 * rank))), 0)
        target_stock_level = reorder_point + forecast_units / forecast_days * lead_time_days
        order_quantity = max(int(round(target_stock_level - simulated_current_stock)), 0)
        days_of_cover = simulated_current_stock / avg_daily_units if avg_daily_units > 0 else np.nan
        risk_flag = "Understock" if simulated_current_stock < reorder_point else "Overstock" if simulated_current_stock > target_stock_level * 1.4 else "Healthy"
        rows.append(
            {
                "StockCode": row["StockCode"],
                "Description": row["Description"],
                "total_units": float(row["total_units"]),
                "mean_daily_units": avg_daily_units,
                "std_daily_units": demand_std,
                "demand_share": demand_share,
                "forecast_units_30d": forecast_units,
                "current_stock": simulated_current_stock,
                "safety_stock": float(safety_stock),
                "reorder_point": float(reorder_point),
                "target_stock_level": float(target_stock_level),
                "recommended_order_qty": order_quantity,
                "days_of_cover": float(days_of_cover),
                "risk_flag": risk_flag,
                "service_level": service_level,
            }
        )

    recommendations = pd.DataFrame(rows).sort_values(["risk_flag", "recommended_order_qty"], ascending=[True, False]).reset_index(drop=True)
    alerts = recommendations.loc[recommendations["risk_flag"] != "Healthy"].copy()

    aggregate_schedule = ensemble[["date", "prophet_quantity_forecast", "lstm_quantity_forecast", "ensemble_quantity_forecast"]].copy()
    aggregate_schedule["prophet_quantity_forecast"] = prophet["yhat"].head(len(aggregate_schedule)).to_numpy() / max(average_unit_price, 1e-6)
    aggregate_schedule["consensus_units"] = aggregate_schedule[["prophet_quantity_forecast", "lstm_quantity_forecast", "ensemble_quantity_forecast"]].mean(axis=1)
    aggregate_schedule["forecast_quantity"] = aggregate_schedule["ensemble_quantity_forecast"]
    return recommendations, alerts, aggregate_schedule


def save_stock_risk_chart(recommendations: pd.DataFrame, output_path: Path) -> Path:
    top = recommendations.head(10).copy()
    fig, ax = plt.subplots(figsize=(15, 7))
    indices = np.arange(len(top))
    ax.bar(indices - 0.2, top["current_stock"], width=0.2, label="Current Stock", color="#1f77b4")
    ax.bar(indices, top["reorder_point"], width=0.2, label="Reorder Point", color="#ff7f0e")
    ax.bar(indices + 0.2, top["target_stock_level"], width=0.2, label="Target Stock", color="#2ca02c")
    ax.set_xticks(indices)
    ax.set_xticklabels(top["StockCode"], rotation=45, ha="right")
    ax.set_title("Inventory Stock Risk Overview")
    ax.set_ylabel("Units")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def save_reorder_schedule_chart(schedule: pd.DataFrame, output_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(schedule["date"], schedule["forecast_quantity"], label="Ensemble Forecast", color="#d62728", linewidth=2)
    ax.plot(schedule["date"], schedule["consensus_units"], label="Consensus Units", color="#1f77b4", linestyle="--")
    ax.set_title("Aggregate Reorder Schedule")
    ax.set_xlabel("Date")
    ax.set_ylabel("Forecast Units")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def save_dashboard_chart(recommendations: pd.DataFrame, alerts: pd.DataFrame, schedule: pd.DataFrame, output_path: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    axes[0].bar(recommendations["risk_flag"].value_counts().index.astype(str), recommendations["risk_flag"].value_counts().values, color=["#2ca02c", "#ff7f0e", "#d62728"])
    axes[0].set_title("Risk Distribution")

    axes[1].hist(recommendations["days_of_cover"].dropna(), bins=10, color="#1f77b4", alpha=0.8)
    axes[1].set_title("Days of Cover")

    axes[2].plot(schedule["date"], schedule["forecast_quantity"], color="#d62728", label="Forecast")
    axes[2].set_title("Forecast Demand Curve")
    axes[2].legend()

    axes[3].bar(["Alerts", "Healthy"], [len(alerts), len(recommendations) - len(alerts)], color=["#d62728", "#2ca02c"])
    axes[3].set_title("Alert Snapshot")

    fig.suptitle("RetailPulse Inventory Health Dashboard", fontsize=16)
    plt.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def build_report(recommendations: pd.DataFrame, alerts: pd.DataFrame, report_path: Path) -> None:
    def _md(frame: pd.DataFrame) -> str:
        try:
            return frame.to_markdown(index=False)
        except ImportError:
            return frame.to_csv(index=False)

    lines = [
        "# RetailPulse Inventory Optimization Report",
        "",
        "## Summary",
        "",
        f"Recommended actions were generated for {len(recommendations)} key SKUs.",
        f"{len(alerts)} SKU(s) require immediate attention due to understock or overstock risk.",
        "",
        "## Reorder Recommendations",
        "",
        _md(recommendations[["StockCode", "Description", "current_stock", "reorder_point", "target_stock_level", "recommended_order_qty", "risk_flag"]].head(15)),
        "",
        "## Business Interpretation",
        "",
        "- Reorder point logic ties forecasted demand to realistic lead-time coverage so replenishment is demand-driven rather than reactive.",
        "- Safety stock helps absorb volatility from the ensemble forecast and protects against retail stockouts.",
        "- Understock alerts highlight where the business may lose sales due to insufficient stock coverage.",
        "- Overstock alerts indicate capital is tied up longer than needed and may create markdown risk.",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_inventory_optimization_pipeline() -> InventoryRunResult:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    recommendations, alerts, schedule = build_inventory_recommendations()

    recommendations_path = PROCESSED_DIR / "inventory_recommendations.csv"
    alerts_path = PROCESSED_DIR / "inventory_alerts.csv"
    metrics_path = PROCESSED_DIR / "inventory_metrics.csv"
    report_path = PROCESSED_DIR / "inventory_optimization_report.md"
    stock_risk_chart_path = FIGURES_DIR / "inventory_stock_risk.png"
    reorder_schedule_chart_path = FIGURES_DIR / "inventory_reorder_schedule.png"
    dashboard_chart_path = FIGURES_DIR / "inventory_health_dashboard.png"

    recommendations.to_csv(recommendations_path, index=False)
    alerts.to_csv(alerts_path, index=False)
    metrics = pd.DataFrame(
        [
            {
                "top_sku_count": int(len(recommendations)),
                "alert_count": int(len(alerts)),
                "understock_count": int((recommendations["risk_flag"] == "Understock").sum()),
                "overstock_count": int((recommendations["risk_flag"] == "Overstock").sum()),
                "avg_days_of_cover": float(recommendations["days_of_cover"].mean()),
                "avg_order_qty": float(recommendations["recommended_order_qty"].mean()),
            }
        ]
    )
    metrics.to_csv(metrics_path, index=False)

    save_stock_risk_chart(recommendations, stock_risk_chart_path)
    save_reorder_schedule_chart(schedule, reorder_schedule_chart_path)
    save_dashboard_chart(recommendations, alerts, schedule, dashboard_chart_path)
    build_report(recommendations, alerts, report_path)

    summary_path = PROCESSED_DIR / "inventory_streamlit_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "recommendations_csv": str(recommendations_path),
                "alerts_csv": str(alerts_path),
                "metrics_csv": str(metrics_path),
                "report": str(report_path),
                "charts": [str(stock_risk_chart_path), str(reorder_schedule_chart_path), str(dashboard_chart_path)],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return InventoryRunResult(
        recommendations_path=recommendations_path,
        alerts_path=alerts_path,
        metrics_path=metrics_path,
        report_path=report_path,
        stock_risk_chart_path=stock_risk_chart_path,
        reorder_schedule_chart_path=reorder_schedule_chart_path,
        dashboard_chart_path=dashboard_chart_path,
    )


if __name__ == "__main__":
    run_inventory_optimization_pipeline()
