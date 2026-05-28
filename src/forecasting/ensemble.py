from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

from retailpulse_lstm_pipeline import RetailDemandLSTM
from retailpulse_mlflow_utils import log_dataframe_artifact, log_json_artifact, log_text_artifact, safe_register_model, setup_mlflow, start_mlflow_run

ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "processed"
MODELS_DIR = PROCESSED_DIR / "models"
FIGURES_DIR = PROCESSED_DIR / "figures"


@dataclass
class HybridForecastResult:
    evaluation_path: Path
    future_forecast_path: Path
    model_path: Path
    metrics_path: Path
    report_path: Path
    comparison_chart_path: Path
    residual_chart_path: Path
    error_chart_path: Path
    metrics: dict[str, float]
    weights: dict[str, float]


class HybridForecastPyFuncModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        with open(context.artifacts["prophet_model"], "rb") as handle:
            self.prophet_model = pickle.load(handle)
        self.lstm_state_dict = torch.load(context.artifacts["lstm_state_dict"], map_location="cpu")
        with open(context.artifacts["lstm_config"], "r", encoding="utf-8") as handle:
            self.lstm_config = json.load(handle)
        with open(context.artifacts["lstm_scaler"], "rb") as handle:
            self.scaler = pickle.load(handle)
        with open(context.artifacts["ensemble_config"], "r", encoding="utf-8") as handle:
            self.ensemble_config = json.load(handle)

        self.lstm_model = RetailDemandLSTM(
            input_size=self.lstm_config["input_size"],
            hidden_size=self.lstm_config["hidden_size"],
            num_layers=self.lstm_config["num_layers"],
            dropout=self.lstm_config["dropout"],
            learning_rate=self.lstm_config["learning_rate"],
        )
        self.lstm_model.load_state_dict(self.lstm_state_dict)
        self.lstm_model.eval()

    def predict(self, context, model_input):
        frame = pd.DataFrame(model_input).copy()
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.sort_values("date")
        if "total_price" not in frame.columns:
            raise ValueError("Input data must include historical 'total_price' values.")

        horizon = int(self.ensemble_config["forecast_horizon"])
        sequence_length = int(self.ensemble_config["sequence_length"])
        weights = self.ensemble_config["weights"]

        prophet_future = self.prophet_model.make_future_dataframe(periods=horizon, freq="D")
        prophet_forecast = self.prophet_model.predict(prophet_future).tail(horizon).reset_index(drop=True)

        history_values = frame["total_price"].astype(float).to_numpy()
        lstm_future = recursive_lstm_forecast(self.lstm_model, self.scaler, history_values, horizon=horizon, sequence_length=sequence_length)

        combined = pd.DataFrame({
            "ds": pd.to_datetime(prophet_forecast["ds"]),
            "prophet_yhat": prophet_forecast["yhat"].to_numpy(dtype=float),
            "lstm_yhat": lstm_future["lstm_yhat"].to_numpy(dtype=float),
        })
        combined["ensemble_yhat"] = (
            weights["prophet"] * combined["prophet_yhat"] + weights["lstm"] * combined["lstm_yhat"]
        )
        avg_unit_price = float(self.ensemble_config["avg_unit_price"])
        combined["forecast_quantity"] = combined["ensemble_yhat"] / avg_unit_price
        return combined


def load_daily_sales() -> pd.DataFrame:
    frame = pd.read_csv(PROCESSED_DIR / "daily_sales_forecasting.csv", parse_dates=["date"])
    frame = frame[["date", "total_price", "quantity"]].dropna().sort_values("date").reset_index(drop=True)
    return frame


def load_prophet_validation_forecast() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "prophet_validation_forecast.csv", parse_dates=["ds"])


def load_prophet_model():
    with open(MODELS_DIR / "prophet_model.pkl", "rb") as handle:
        return pickle.load(handle)


def load_lstm_artifacts() -> tuple[RetailDemandLSTM, dict[str, Any], Any]:
    with open(MODELS_DIR / "lstm_model_state_dict.pt", "rb") as handle:
        state_dict = torch.load(handle, map_location="cpu")
    with open(MODELS_DIR / "lstm_config.json", "r", encoding="utf-8") as handle:
        config = json.load(handle)
    with open(MODELS_DIR / "lstm_scaler.pkl", "rb") as handle:
        scaler = pickle.load(handle)

    model = RetailDemandLSTM(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        learning_rate=config["learning_rate"],
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model, config, scaler


def recursive_lstm_forecast(model: RetailDemandLSTM, scaler, history_values: np.ndarray, horizon: int, sequence_length: int) -> pd.DataFrame:
    history_values = np.asarray(history_values, dtype=float)
    scaled_history = scaler.transform(history_values.reshape(-1, 1)).reshape(-1)
    current_window = scaled_history[-sequence_length:].copy()
    predictions_scaled: list[float] = []

    with torch.no_grad():
        for _ in range(horizon):
            input_tensor = torch.tensor(current_window.reshape(1, sequence_length, 1), dtype=torch.float32)
            next_prediction = float(model(input_tensor).cpu().numpy().ravel()[0])
            predictions_scaled.append(next_prediction)
            current_window = np.append(current_window[1:], next_prediction)

    predictions = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1)).reshape(-1)
    return pd.DataFrame({"lstm_yhat": predictions})


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    actual = pd.to_numeric(y_true, errors="coerce").astype(float).to_numpy()
    predicted = pd.to_numeric(y_pred, errors="coerce").astype(float).to_numpy()
    denominator = np.clip(np.abs(actual), 1e-8, None)
    return {
        "MAPE": float(np.mean(np.abs((actual - predicted) / denominator)) * 100),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "MAE": float(mean_absolute_error(actual, predicted)),
    }


def normalize_weights(configured_weights: dict[str, float] | None, metrics: dict[str, dict[str, float]]) -> dict[str, float]:
    if configured_weights:
        total = sum(configured_weights.values()) or 1.0
        return {name: value / total for name, value in configured_weights.items()}

    prophet_score = 1.0 / max(metrics["prophet"]["RMSE"], 1e-6)
    lstm_score = 1.0 / max(metrics["lstm"]["RMSE"], 1e-6)
    total = prophet_score + lstm_score
    return {"prophet": prophet_score / total, "lstm": lstm_score / total}


def build_evaluation_frames(horizon: int = 30) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sales = load_daily_sales()
    actual_window = sales.tail(horizon).copy().reset_index(drop=True)
    prophet_validation = load_prophet_validation_forecast().rename(columns={"ds": "date"}).copy()
    prophet_validation = prophet_validation[["date", "yhat", "yhat_lower", "yhat_upper"]].rename(columns={"yhat": "prophet_yhat"})
    prophet_window = prophet_validation.merge(actual_window[["date", "total_price"]], on="date", how="inner")

    lstm_model, lstm_config, scaler = load_lstm_artifacts()
    lstm_window = recursive_lstm_forecast(
        lstm_model,
        scaler,
        sales.iloc[:-horizon]["total_price"].to_numpy(),
        horizon=horizon,
        sequence_length=int(lstm_config["sequence_length"]),
    )
    lstm_window["date"] = actual_window["date"].values
    lstm_window = lstm_window[["date", "lstm_yhat"]]

    combined = actual_window[["date", "total_price"]].merge(prophet_window[["date", "prophet_yhat"]], on="date", how="left").merge(lstm_window, on="date", how="left")
    combined = combined.dropna(subset=["prophet_yhat", "lstm_yhat"]).reset_index(drop=True)

    prophet_metrics = compute_metrics(combined["total_price"], combined["prophet_yhat"])
    lstm_metrics = compute_metrics(combined["total_price"], combined["lstm_yhat"])
    weights = normalize_weights(None, {"prophet": prophet_metrics, "lstm": lstm_metrics})
    combined["ensemble_yhat"] = weights["prophet"] * combined["prophet_yhat"] + weights["lstm"] * combined["lstm_yhat"]
    return combined, actual_window, pd.DataFrame([prophet_metrics | {"model": "Prophet"}, lstm_metrics | {"model": "LSTM"}]), pd.DataFrame([weights])


def forecast_future(horizon: int = 30, weights: dict[str, float] | None = None) -> pd.DataFrame:
    sales = load_daily_sales()
    prophet_model = load_prophet_model()
    lstm_model, lstm_config, scaler = load_lstm_artifacts()

    prophet_future = prophet_model.make_future_dataframe(periods=horizon, freq="D")
    prophet_forecast = prophet_model.predict(prophet_future).tail(horizon).reset_index(drop=True)

    lstm_forecast = recursive_lstm_forecast(
        lstm_model,
        scaler,
        sales["total_price"].to_numpy(),
        horizon=horizon,
        sequence_length=int(lstm_config["sequence_length"]),
    )

    avg_unit_price = float(max(sales["total_price"].sum() / max(sales["quantity"].sum(), 1e-6), 1e-6))
    weights = weights or {"prophet": 0.5, "lstm": 0.5}
    future = pd.DataFrame({
        "date": pd.to_datetime(prophet_forecast["ds"]),
        "prophet_yhat": prophet_forecast["yhat"].to_numpy(dtype=float),
        "prophet_yhat_lower": prophet_forecast["yhat_lower"].to_numpy(dtype=float),
        "prophet_yhat_upper": prophet_forecast["yhat_upper"].to_numpy(dtype=float),
        "lstm_yhat": lstm_forecast["lstm_yhat"].to_numpy(dtype=float),
    })
    future["ensemble_yhat"] = weights["prophet"] * future["prophet_yhat"] + weights["lstm"] * future["lstm_yhat"]
    future["prophet_quantity_forecast"] = future["prophet_yhat"] / avg_unit_price
    future["lstm_quantity_forecast"] = future["lstm_yhat"] / avg_unit_price
    future["ensemble_quantity_forecast"] = future["ensemble_yhat"] / avg_unit_price
    future["forecast_quantity"] = future["ensemble_quantity_forecast"]
    future["forecast_lower_quantity"] = future["prophet_yhat_lower"] / avg_unit_price
    future["forecast_upper_quantity"] = future["prophet_yhat_upper"] / avg_unit_price
    return future


def save_comparison_chart(frame: pd.DataFrame, output_path: Path) -> Path:
    plt.figure(figsize=(15, 7))
    plt.plot(frame["date"], frame["total_price"], label="Actual", color="#1f77b4", linewidth=2)
    plt.plot(frame["date"], frame["prophet_yhat"], label="Prophet", color="#ff7f0e", linewidth=2)
    plt.plot(frame["date"], frame["lstm_yhat"], label="LSTM", color="#2ca02c", linewidth=2)
    plt.plot(frame["date"], frame["ensemble_yhat"], label="Ensemble", color="#d62728", linewidth=2.5)
    plt.title("Day 8 Hybrid Forecast Comparison")
    plt.xlabel("Date")
    plt.ylabel("Revenue")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def save_residual_chart(frame: pd.DataFrame, output_path: Path) -> Path:
    residuals = pd.DataFrame({
        "date": frame["date"],
        "prophet_residual": frame["total_price"] - frame["prophet_yhat"],
        "lstm_residual": frame["total_price"] - frame["lstm_yhat"],
        "ensemble_residual": frame["total_price"] - frame["ensemble_yhat"],
    })
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=False)
    axes[0].plot(residuals["date"], residuals["prophet_residual"], label="Prophet", color="#ff7f0e")
    axes[0].plot(residuals["date"], residuals["lstm_residual"], label="LSTM", color="#2ca02c")
    axes[0].plot(residuals["date"], residuals["ensemble_residual"], label="Ensemble", color="#d62728")
    axes[0].axhline(0, color="#111827", linestyle="--", linewidth=1)
    axes[0].set_title("Residuals Over Time")
    axes[0].set_ylabel("Actual - Forecast")
    axes[0].legend()
    axes[1].hist([residuals["prophet_residual"], residuals["lstm_residual"], residuals["ensemble_residual"]], bins=12, label=["Prophet", "LSTM", "Ensemble"], color=["#ff7f0e", "#2ca02c", "#d62728"], alpha=0.65)
    axes[1].set_title("Residual Distribution")
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def save_error_chart(frame: pd.DataFrame, output_path: Path) -> Path:
    errors = pd.DataFrame({
        "Prophet": (frame["total_price"] - frame["prophet_yhat"]).abs(),
        "LSTM": (frame["total_price"] - frame["lstm_yhat"]).abs(),
        "Ensemble": (frame["total_price"] - frame["ensemble_yhat"]).abs(),
    })
    plt.figure(figsize=(14, 6))
    plt.hist([errors["Prophet"], errors["LSTM"], errors["Ensemble"]], bins=12, label=["Prophet", "LSTM", "Ensemble"], color=["#ff7f0e", "#2ca02c", "#d62728"], alpha=0.7)
    plt.title("Absolute Error Distribution")
    plt.xlabel("Absolute Error")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def build_report(evaluation: pd.DataFrame, metrics: pd.DataFrame, weights: dict[str, float], report_path: Path, future_forecast: pd.DataFrame) -> None:
    future_summary = future_forecast[["date", "prophet_yhat", "lstm_yhat", "ensemble_yhat", "prophet_quantity_forecast", "lstm_quantity_forecast", "ensemble_quantity_forecast"]].head(10).copy()
    future_summary["date"] = pd.to_datetime(future_summary["date"]).dt.strftime("%Y-%m-%d")
    lines = [
        "# RetailPulse Hybrid Forecast Ensemble Report",
        "",
        "## Ensemble Weights",
        "",
        pd.DataFrame([weights]).to_markdown(index=False),
        "",
        "## Model Comparison",
        "",
        metrics.to_markdown(index=False),
        "",
        "## Evaluation Window",
        "",
        evaluation[["date", "total_price", "prophet_yhat", "lstm_yhat", "ensemble_yhat"]].to_markdown(index=False),
        "",
        "## Future Forecast Summary",
        "",
        future_summary.to_markdown(index=False),
        "",
        "## Business Interpretation",
        "",
        "- The ensemble is intended to reduce single-model volatility by blending Prophet trend stability with LSTM nonlinear sequence sensitivity.",
        "- If the ensemble outperforms both base models on RMSE and MAE, it is the preferred planning signal for retail demand decisions.",
        "- A stable forecast curve with controlled residual spread indicates stronger planning reliability for inventory and operations.",
        "- Forecast uncertainty should be treated as a planning band, not a promise, especially around promotions or calendar shocks.",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_hybrid_forecasting_ensemble(horizon: int = 30, configured_weights: dict[str, float] | None = None) -> HybridForecastResult:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    evaluation, actual_window, model_metrics, derived_weights = build_evaluation_frames(horizon=horizon)
    weights = normalize_weights(configured_weights, {row["model"].lower(): {"RMSE": row["RMSE"]} for _, row in model_metrics.iterrows()}) if configured_weights else derived_weights.iloc[0].to_dict()
    if configured_weights:
        total = sum(configured_weights.values()) or 1.0
        weights = {name: value / total for name, value in configured_weights.items()}

    evaluation["ensemble_yhat"] = weights["prophet"] * evaluation["prophet_yhat"] + weights["lstm"] * evaluation["lstm_yhat"]
    evaluation["prophet_residual"] = evaluation["total_price"] - evaluation["prophet_yhat"]
    evaluation["lstm_residual"] = evaluation["total_price"] - evaluation["lstm_yhat"]
    evaluation["ensemble_residual"] = evaluation["total_price"] - evaluation["ensemble_yhat"]

    evaluation_metrics = pd.DataFrame([
        {"model": "Prophet", **compute_metrics(evaluation["total_price"], evaluation["prophet_yhat"])},
        {"model": "LSTM", **compute_metrics(evaluation["total_price"], evaluation["lstm_yhat"])},
        {"model": "Ensemble", **compute_metrics(evaluation["total_price"], evaluation["ensemble_yhat"])},
    ])
    metrics_summary = {
        f"{row['model'].lower()}_{metric.lower()}": float(row[metric])
        for _, row in evaluation_metrics.iterrows()
        for metric in ["MAPE", "RMSE", "MAE"]
    }

    future_forecast = forecast_future(horizon=horizon, weights=weights)

    evaluation_path = PROCESSED_DIR / "ensemble_evaluation_forecast.csv"
    future_forecast_path = PROCESSED_DIR / "ensemble_future_30d_forecast.csv"
    lstm_future_path = PROCESSED_DIR / "lstm_future_30d_forecast.csv"
    metrics_path = PROCESSED_DIR / "ensemble_metrics.csv"
    model_path = MODELS_DIR / "ensemble_model.pkl"
    report_path = PROCESSED_DIR / "ensemble_forecasting_report.md"
    config_path = PROCESSED_DIR / "ensemble_config.json"
    comparison_chart_path = FIGURES_DIR / "ensemble_forecast_comparison.png"
    residual_chart_path = FIGURES_DIR / "ensemble_residual_analysis.png"
    error_chart_path = FIGURES_DIR / "ensemble_error_distribution.png"

    evaluation.to_csv(evaluation_path, index=False)
    future_forecast.to_csv(future_forecast_path, index=False)
    future_forecast[["date", "lstm_yhat", "lstm_quantity_forecast"]].to_csv(lstm_future_path, index=False)
    evaluation_metrics.to_csv(metrics_path, index=False)

    ensemble_payload = {
        "weights": weights,
        "forecast_horizon": horizon,
        "sequence_length": 30,
        "metrics": metrics_summary,
        "future_forecast_path": str(future_forecast_path),
    }
    with open(model_path, "wb") as handle:
        pickle.dump(ensemble_payload, handle)
    config_payload = {
        "weights": weights,
        "forecast_horizon": horizon,
        "sequence_length": 30,
        "avg_unit_price": float(max(load_daily_sales()["total_price"].sum() / max(load_daily_sales()["quantity"].sum(), 1e-6), 1e-6)),
    }
    config_path.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    save_comparison_chart(evaluation, comparison_chart_path)
    save_residual_chart(evaluation, residual_chart_path)
    save_error_chart(evaluation, error_chart_path)
    build_report(evaluation, evaluation_metrics, weights, report_path, future_forecast)

    setup_mlflow("RetailPulse")
    with start_mlflow_run("day8_hybrid_ensemble") as run:
        mlflow.log_params({"forecast_horizon": horizon, "prophet_weight": weights["prophet"], "lstm_weight": weights["lstm"]})
        for key, value in metrics_summary.items():
            mlflow.log_metric(key, value)
        log_json_artifact({"weights": weights, "metrics": metrics_summary, "forecast_horizon": horizon}, "ensemble_config.json")
        log_dataframe_artifact(evaluation_metrics, "ensemble_metrics.csv")
        log_dataframe_artifact(evaluation, "ensemble_evaluation_forecast.csv")
        log_dataframe_artifact(future_forecast, "ensemble_future_30d_forecast.csv")
        log_dataframe_artifact(future_forecast[["date", "lstm_yhat", "lstm_quantity_forecast"]], "lstm_future_30d_forecast.csv")
        log_text_artifact(report_path.read_text(encoding="utf-8"), "ensemble_forecasting_report.md")

        mlflow.pyfunc.log_model(
            artifact_path="ensemble_model",
            python_model=HybridForecastPyFuncModel(),
            artifacts={
                "prophet_model": str(MODELS_DIR / "prophet_model.pkl"),
                "lstm_state_dict": str(MODELS_DIR / "lstm_model_state_dict.pt"),
                "lstm_scaler": str(MODELS_DIR / "lstm_scaler.pkl"),
                "lstm_config": str(MODELS_DIR / "lstm_config.json"),
                "ensemble_config": str(config_path),
            },
        )
        safe_register_model(f"runs:/{run.info.run_id}/ensemble_model", "RetailPulseHybridForecastEnsemble")

    return HybridForecastResult(
        evaluation_path=evaluation_path,
        future_forecast_path=future_forecast_path,
        model_path=model_path,
        metrics_path=metrics_path,
        report_path=report_path,
        comparison_chart_path=comparison_chart_path,
        residual_chart_path=residual_chart_path,
        error_chart_path=error_chart_path,
        metrics={k: float(v) for k, v in metrics_summary.items()},
        weights=weights,
    )


if __name__ == "__main__":
    run_hybrid_forecasting_ensemble()
