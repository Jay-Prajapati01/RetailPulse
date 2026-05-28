from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.pyfunc
import mlflow.pytorch
import mlflow.sklearn
import pandas as pd
import torch

from retailpulse_feature_pipeline import MODELS_DIR, PROCESSED_DIR, FIGURES_DIR, ensure_output_dirs
from retailpulse_lstm_pipeline import RetailDemandLSTM
from retailpulse_mlflow_utils import DEFAULT_EXPERIMENT, ProphetPyFuncModel, log_dataframe_artifact, log_json_artifact, log_text_artifact, safe_register_model, setup_mlflow


TRACKING_SUMMARY = PROCESSED_DIR / "mlflow_tracking_summary.md"


def load_metrics(file_name: str) -> dict[str, float]:
    frame = pd.read_csv(PROCESSED_DIR / file_name)
    return frame.iloc[0].to_dict()


def log_prophet_experiment() -> dict[str, str]:
    model_path = MODELS_DIR / "prophet_model.pkl"
    metrics = load_metrics("prophet_metrics.csv")
    validation = pd.read_csv(PROCESSED_DIR / "prophet_validation_forecast.csv")
    future = pd.read_csv(PROCESSED_DIR / "prophet_future_30d_forecast.csv")
    params = pd.read_csv(PROCESSED_DIR / "prophet_grid_search_results.csv").sort_values("MAPE").iloc[0].to_dict()

    with mlflow.start_run(run_name="prophet_forecast_tracking"):
        mlflow.log_params({k: str(v) for k, v in params.items() if k not in {"MAPE", "RMSE", "MAE"}})
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
        mlflow.log_artifact(str(PROCESSED_DIR / "prophet_performance_report.md"))
        mlflow.log_artifact(str(PROCESSED_DIR / "prophet_grid_search_results.csv"))
        mlflow.log_artifact(str(PROCESSED_DIR / "prophet_validation_forecast.csv"))
        mlflow.log_artifact(str(PROCESSED_DIR / "prophet_future_30d_forecast.csv"))
        model_info = mlflow.pyfunc.log_model(
            artifact_path="prophet_model",
            python_model=ProphetPyFuncModel(),
            artifacts={"prophet_model": str(model_path)},
            input_example=validation[["ds"]].head(5),
        )
        registration = safe_register_model(model_info.model_uri, "RetailPulseProphetForecast")
        return {"run": "prophet", "model_uri": model_info.model_uri, **registration}


def log_lstm_experiment() -> dict[str, str]:
    metrics = load_metrics("lstm_metrics.csv")
    config = json.loads((MODELS_DIR / "lstm_config.json").read_text(encoding="utf-8"))
    scaler_path = MODELS_DIR / "lstm_scaler.pkl"
    state_dict_path = MODELS_DIR / "lstm_model_state_dict.pt"
    model = RetailDemandLSTM(
        input_size=int(config["input_size"]),
        hidden_size=int(config["hidden_size"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config["dropout"]),
        learning_rate=float(config["learning_rate"]),
    )
    state_dict = torch.load(state_dict_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    with mlflow.start_run(run_name="lstm_forecast_tracking"):
        mlflow.log_params(config)
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
        mlflow.log_artifact(str(PROCESSED_DIR / "lstm_evaluation_report.md"))
        mlflow.log_artifact(str(PROCESSED_DIR / "lstm_test_predictions.csv"))
        mlflow.log_artifact(str(MODELS_DIR / "lstm_config.json"))
        mlflow.log_artifact(str(MODELS_DIR / "lstm_scaler.pkl"))
        model_info = mlflow.pytorch.log_model(model, artifact_path="lstm_model")
        registration = safe_register_model(model_info.model_uri, "RetailPulseLSTMForecast")
        return {"run": "lstm", "model_uri": model_info.model_uri, **registration}


def log_clustering_experiment() -> dict[str, str]:
    kmeans_model = joblib.load(MODELS_DIR / "customer_kmeans_model.joblib")
    dbscan_model = joblib.load(MODELS_DIR / "customer_dbscan_model.joblib")
    cluster_profile = pd.read_csv(PROCESSED_DIR / "cluster_profile_kmeans.csv")
    cluster_metrics = pd.read_csv(PROCESSED_DIR / "kmeans_elbow_metrics.csv")
    dbscan_metrics = pd.read_csv(PROCESSED_DIR / "dbscan_grid_metrics.csv")
    cluster_labels = pd.read_csv(PROCESSED_DIR / "customer_cluster_labels.csv")

    with mlflow.start_run(run_name="customer_clustering_tracking"):
        mlflow.log_params({"kmeans_clusters": int(kmeans_model.n_clusters), "dbscan_eps": float(dbscan_model.eps), "dbscan_min_samples": int(dbscan_model.min_samples)})
        mlflow.log_metrics({
            "kmeans_silhouette_best": float(cluster_metrics.sort_values("silhouette", ascending=False).iloc[0]["silhouette"]),
            "kmeans_inertia_best": float(cluster_metrics.sort_values("silhouette", ascending=False).iloc[0]["inertia"]),
            "dbscan_silhouette_best": float(pd.to_numeric(dbscan_metrics["silhouette"], errors="coerce").max()),
        })
        mlflow.log_artifact(str(PROCESSED_DIR / "customer_segmentation_report.md"))
        mlflow.log_artifact(str(PROCESSED_DIR / "customer_personas.csv"))
        mlflow.log_artifact(str(PROCESSED_DIR / "customer_cluster_labels.csv"))
        mlflow.log_artifact(str(PROCESSED_DIR / "cluster_profile_kmeans.csv"))
        mlflow.log_artifact(str(PROCESSED_DIR / "cluster_profile_dbscan.csv"))
        mlflow.log_artifact(str(FIGURES_DIR / "kmeans_pca_clusters.png"))
        mlflow.log_artifact(str(FIGURES_DIR / "dbscan_pca_clusters.png"))
        mlflow.log_artifact(str(FIGURES_DIR / "kmeans_tsne_clusters.png"))
        model_info = mlflow.sklearn.log_model(kmeans_model, artifact_path="customer_kmeans_model")
        registration = safe_register_model(model_info.model_uri, "RetailPulseCustomerKMeans")
        return {"run": "clustering", "model_uri": model_info.model_uri, **registration}


def write_tracking_summary(records: list[dict[str, str]]) -> Path:
    lines = [
        "# RetailPulse MLflow Tracking Summary",
        "",
        "This file records the local MLflow experiment runs created for the project.",
        "",
    ]
    for record in records:
        lines.append(f"- {record['run']}: {record.get('model_uri', 'n/a')}")
        if "registered_version" in record:
            lines.append(f"  - registered_version: {record['registered_version']}")
        if "registration_error" in record:
            lines.append(f"  - registration_error: {record['registration_error']}")
    TRACKING_SUMMARY.write_text("\n".join(lines), encoding="utf-8")
    return TRACKING_SUMMARY


def run_mlflow_tracking() -> list[dict[str, str]]:
    ensure_output_dirs()
    setup_mlflow(DEFAULT_EXPERIMENT)
    records = [log_prophet_experiment(), log_lstm_experiment(), log_clustering_experiment()]
    write_tracking_summary(records)
    return records


if __name__ == "__main__":
    run_mlflow_tracking()
