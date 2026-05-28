from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

from retailpulse_feature_pipeline import FIGURES_DIR, MODELS_DIR, PROCESSED_DIR, ensure_output_dirs


@dataclass
class ProphetRunResult:
    best_params: dict[str, Any]
    metrics: dict[str, float]
    validation_forecast_path: Path
    future_forecast_path: Path
    full_forecast_path: Path
    model_path: Path
    report_path: Path


def load_daily_sales() -> pd.DataFrame:
    frame = pd.read_csv(PROCESSED_DIR / "daily_sales_forecasting.csv", parse_dates=["date"])
    frame = frame[["date", "total_price"]].dropna().sort_values("date").reset_index(drop=True)
    return frame


def prepare_prophet_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prophet_frame = frame.rename(columns={"date": "ds", "total_price": "y"}).copy()
    prophet_frame["ds"] = pd.to_datetime(prophet_frame["ds"])
    return prophet_frame


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    actual = pd.to_numeric(y_true, errors="coerce").astype(float).to_numpy()
    predicted = pd.to_numeric(y_pred, errors="coerce").astype(float).to_numpy()
    denominator = np.clip(np.abs(actual), 1e-8, None)
    mape = float(np.mean(np.abs((actual - predicted) / denominator)) * 100)
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    mae = float(mean_absolute_error(actual, predicted))
    return {"MAPE": mape, "RMSE": rmse, "MAE": mae}


def build_forecast_model(config: dict[str, Any]) -> Prophet:
    return Prophet(
        growth="linear",
        yearly_seasonality=config["yearly_seasonality"],
        weekly_seasonality=config["weekly_seasonality"],
        daily_seasonality=False,
        seasonality_mode=config["seasonality_mode"],
        seasonality_prior_scale=config["seasonality_prior_scale"],
        changepoint_prior_scale=config["changepoint_prior_scale"],
        interval_width=0.95,
    )


def tune_prophet_model(train_df: pd.DataFrame, validation_df: pd.DataFrame) -> tuple[Prophet, dict[str, Any], pd.DataFrame, list[dict[str, Any]]]:
    grid = [
        {"yearly_seasonality": True, "weekly_seasonality": True, "seasonality_mode": "additive", "seasonality_prior_scale": 10.0, "changepoint_prior_scale": 0.05},
        {"yearly_seasonality": True, "weekly_seasonality": True, "seasonality_mode": "multiplicative", "seasonality_prior_scale": 10.0, "changepoint_prior_scale": 0.05},
        {"yearly_seasonality": True, "weekly_seasonality": False, "seasonality_mode": "additive", "seasonality_prior_scale": 5.0, "changepoint_prior_scale": 0.1},
        {"yearly_seasonality": True, "weekly_seasonality": False, "seasonality_mode": "multiplicative", "seasonality_prior_scale": 5.0, "changepoint_prior_scale": 0.1},
    ]
    evaluation_rows: list[dict[str, Any]] = []
    best_model = None
    best_params: dict[str, Any] = {}
    best_metrics: dict[str, float] = {}
    best_validation = None
    best_score = float("inf")

    for config in grid:
        model = build_forecast_model(config)
        model.fit(train_df)
        forecast = model.predict(validation_df[["ds"]])[ ["ds", "yhat", "yhat_lower", "yhat_upper"] ]
        merged = validation_df.merge(forecast, on="ds", how="left")
        metrics = compute_metrics(merged["y"], merged["yhat"])
        row = {**config, **metrics}
        evaluation_rows.append(row)
        if metrics["MAPE"] < best_score:
            best_score = metrics["MAPE"]
            best_model = model
            best_params = config
            best_metrics = metrics
            best_validation = merged.copy()

    assert best_model is not None and best_validation is not None
    return best_model, best_params, best_validation, evaluation_rows


def plot_validation_forecast(validation_df: pd.DataFrame) -> Path:
    plt.figure(figsize=(14, 6))
    plt.plot(validation_df["ds"], validation_df["y"], label="Actual", color="#1f77b4", linewidth=2)
    plt.plot(validation_df["ds"], validation_df["yhat"], label="Forecast", color="#d62728", linewidth=2)
    plt.fill_between(validation_df["ds"], validation_df["yhat_lower"], validation_df["yhat_upper"], color="#d62728", alpha=0.15, label="95% interval")
    plt.title("Prophet Validation Forecast")
    plt.xlabel("Date")
    plt.ylabel("Revenue")
    plt.legend()
    plt.tight_layout()
    output_path = FIGURES_DIR / "prophet_validation_forecast.png"
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def plot_future_forecast(full_forecast: pd.DataFrame, history_df: pd.DataFrame) -> Path:
    plt.figure(figsize=(14, 6))
    plt.plot(history_df["ds"], history_df["y"], label="History", color="#1f77b4", linewidth=2)
    forecast_future = full_forecast.loc[full_forecast["ds"] > history_df["ds"].max()]
    plt.plot(forecast_future["ds"], forecast_future["yhat"], label="Next 30 Days", color="#2ca02c", linewidth=2)
    plt.fill_between(forecast_future["ds"], forecast_future["yhat_lower"], forecast_future["yhat_upper"], color="#2ca02c", alpha=0.15, label="95% interval")
    plt.title("Prophet 30-Day Revenue Forecast")
    plt.xlabel("Date")
    plt.ylabel("Revenue")
    plt.legend()
    plt.tight_layout()
    output_path = FIGURES_DIR / "prophet_future_forecast.png"
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def build_report(best_params: dict[str, Any], metrics: dict[str, float], eval_table: pd.DataFrame, future_forecast: pd.DataFrame, report_path: Path) -> None:
    top_future = future_forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].head(10).copy()
    top_future["ds"] = pd.to_datetime(top_future["ds"]).dt.strftime("%Y-%m-%d")
    lines = [
        "# RetailPulse Prophet Forecasting Report",
        "",
        "## Best Configuration",
        "",
        pd.DataFrame([best_params]).to_markdown(index=False),
        "",
        "## Validation Metrics",
        "",
        pd.DataFrame([metrics]).to_markdown(index=False),
        "",
        "## Grid Search Summary",
        "",
        eval_table.sort_values("MAPE").to_markdown(index=False),
        "",
        "## Next 30 Days Forecast",
        "",
        top_future.to_markdown(index=False),
        "",
        "## Business Interpretation",
        "",
        "- The best Prophet configuration is selected on the last 30 days of observed sales, which gives an honest out-of-sample test.",
        "- Confidence intervals should be used as planning bands rather than exact revenue promises.",
        "- If weekly seasonality dominates, RetailPulse should align inventory and campaign cadence to day-of-week effects.",
        "- Yearly seasonality becomes more important as the business crosses holiday and seasonal buying cycles.",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_prophet_pipeline() -> ProphetRunResult:
    ensure_output_dirs()
    sales = load_daily_sales()
    prophet_frame = prepare_prophet_frame(sales)

    validation_horizon = 30
    train_df = prophet_frame.iloc[:-validation_horizon].copy()
    validation_df = prophet_frame.iloc[-validation_horizon:].copy()

    best_model, best_params, validation_forecast, grid_rows = tune_prophet_model(train_df, validation_df)
    validation_metrics = compute_metrics(validation_forecast["y"], validation_forecast["yhat"])

    full_model = build_forecast_model(best_params)
    full_model.fit(prophet_frame)

    future = full_model.make_future_dataframe(periods=30, freq="D")
    full_forecast = full_model.predict(future)
    future_forecast = full_forecast.loc[full_forecast["ds"] > prophet_frame["ds"].max()].copy()

    validation_forecast_path = PROCESSED_DIR / "prophet_validation_forecast.csv"
    future_forecast_path = PROCESSED_DIR / "prophet_future_30d_forecast.csv"
    full_forecast_path = PROCESSED_DIR / "prophet_full_forecast.csv"
    metrics_path = PROCESSED_DIR / "prophet_metrics.csv"
    grid_path = PROCESSED_DIR / "prophet_grid_search_results.csv"
    report_path = PROCESSED_DIR / "prophet_performance_report.md"
    model_path = MODELS_DIR / "prophet_model.pkl"

    validation_forecast.to_csv(validation_forecast_path, index=False)
    future_forecast.to_csv(future_forecast_path, index=False)
    full_forecast.to_csv(full_forecast_path, index=False)
    pd.DataFrame([validation_metrics]).to_csv(metrics_path, index=False)
    pd.DataFrame(grid_rows).to_csv(grid_path, index=False)

    with open(model_path, "wb") as handle:
        pickle.dump(full_model, handle)

    plot_validation_forecast(validation_forecast)
    plot_future_forecast(full_forecast, prophet_frame)
    components_figure = full_model.plot_components(full_forecast)
    components_figure.tight_layout()
    components_figure.savefig(FIGURES_DIR / "prophet_components.png", dpi=160)
    plt.close("all")

    build_report(best_params, validation_metrics, pd.DataFrame(grid_rows), future_forecast, report_path)

    return ProphetRunResult(
        best_params=best_params,
        metrics=validation_metrics,
        validation_forecast_path=validation_forecast_path,
        future_forecast_path=future_forecast_path,
        full_forecast_path=full_forecast_path,
        model_path=model_path,
        report_path=report_path,
    )


if __name__ == "__main__":
    run_prophet_pipeline()
