from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator

import json
import pickle

import mlflow
import pandas as pd

from retailpulse_feature_pipeline import MODELS_DIR, PROCESSED_DIR, FIGURES_DIR

ROOT_DIR = Path(__file__).resolve().parent
MLFLOW_DIR = ROOT_DIR / "mlruns"
MLFLOW_STORE_URI = f"file:///{MLFLOW_DIR.as_posix()}"
DEFAULT_EXPERIMENT = "RetailPulse"


def setup_mlflow(experiment_name: str = DEFAULT_EXPERIMENT) -> None:
    MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_STORE_URI)
    mlflow.set_experiment(experiment_name)


@contextmanager
def start_mlflow_run(run_name: str, experiment_name: str = DEFAULT_EXPERIMENT) -> Iterator[Any]:
    setup_mlflow(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        yield run


def log_text_artifact(text: str, artifact_name: str) -> Path:
    output_path = PROCESSED_DIR / artifact_name
    output_path.write_text(text, encoding="utf-8")
    mlflow.log_artifact(str(output_path))
    return output_path


def log_json_artifact(payload: Dict[str, Any], artifact_name: str) -> Path:
    output_path = PROCESSED_DIR / artifact_name
    output_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    mlflow.log_artifact(str(output_path))
    return output_path


def log_dataframe_artifact(frame: pd.DataFrame, artifact_name: str) -> Path:
    output_path = PROCESSED_DIR / artifact_name
    frame.to_csv(output_path, index=False)
    mlflow.log_artifact(str(output_path))
    return output_path


def safe_register_model(model_uri: str, model_name: str) -> dict[str, str]:
    try:
        version = mlflow.register_model(model_uri=model_uri, name=model_name)
        return {"model_name": model_name, "model_uri": model_uri, "registered_version": str(version.version)}
    except Exception as exc:  # pragma: no cover - registration depends on local backend support
        return {"model_name": model_name, "model_uri": model_uri, "registration_error": str(exc)}


class ProphetPyFuncModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        with open(context.artifacts["prophet_model"], "rb") as handle:
            self.model = pickle.load(handle)

    def predict(self, context, model_input):
        frame = model_input.copy()
        if "ds" not in frame.columns:
            if "date" in frame.columns:
                frame["ds"] = pd.to_datetime(frame["date"])
            else:
                frame["ds"] = pd.to_datetime(frame.index)
        forecast = self.model.predict(frame[["ds"]])
        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]


class Paths:
    processed = PROCESSED_DIR
    models = MODELS_DIR
    figures = FIGURES_DIR
