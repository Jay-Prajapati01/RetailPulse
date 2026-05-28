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
import numpy as np
import optuna
from optuna import TrialPruned
import pandas as pd
import joblib
import pytorch_lightning as pl
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from xgboost import XGBClassifier

from retailpulse_feature_pipeline import FIGURES_DIR, MODELS_DIR, PROCESSED_DIR, ensure_output_dirs
from src.forecasting.ensemble import build_evaluation_frames, run_hybrid_forecasting_ensemble
from retailpulse_lstm_pipeline import RetailDemandLSTM, build_sequence_arrays, compute_metrics as lstm_compute_metrics, inverse_transform, load_daily_sales as load_lstm_daily_sales, split_sequences
from retailpulse_mlflow_utils import log_dataframe_artifact, log_json_artifact, log_text_artifact, safe_register_model, setup_mlflow, start_mlflow_run
from src.churn.churn_pipeline import build_churn_dataset, build_feature_matrix, load_customer_base

ROOT_DIR = Path(__file__).resolve().parents[2]
OPTUNA_DIR = PROCESSED_DIR / "optuna"
OPTUNA_MODELS_DIR = MODELS_DIR / "optuna"
OPTUNA_FIGURES_DIR = FIGURES_DIR / "optuna"


@dataclass
class OptimizationResult:
    study_name: str
    best_params: dict[str, Any]
    best_metrics: dict[str, float]
    study_path: Path
    metrics_path: Path
    report_path: Path
    history_chart_path: Path
    importance_chart_path: Path
    model_path: Path | None = None


class LightningPruningCallback(pl.Callback):
    def __init__(self, trial: optuna.Trial, monitor: str = "val_loss"):
        self.trial = trial
        self.monitor = monitor

    def on_validation_epoch_end(self, trainer, pl_module):  # pragma: no cover - callback hook
        metric = trainer.callback_metrics.get(self.monitor)
        if metric is None:
            return
        value = float(metric.detach().cpu().item())
        self.trial.report(value, step=trainer.current_epoch)
        if self.trial.should_prune():
            raise TrialPruned(f"Pruned at epoch {trainer.current_epoch}")


def ensure_optuna_dirs() -> None:
    OPTUNA_DIR.mkdir(parents=True, exist_ok=True)
    OPTUNA_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OPTUNA_FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def save_study_artifacts(study: optuna.Study, study_name: str, metrics_frame: pd.DataFrame, report_lines: list[str]) -> tuple[Path, Path, Path, Path]:
    study_path = OPTUNA_DIR / f"{study_name}_study.json"
    metrics_path = OPTUNA_DIR / f"{study_name}_trial_metrics.csv"
    report_path = OPTUNA_DIR / f"{study_name}_report.md"
    history_chart_path = OPTUNA_FIGURES_DIR / f"{study_name}_history.png"
    importance_chart_path = OPTUNA_FIGURES_DIR / f"{study_name}_importance.png"

    study_path.write_text(study_to_json(study), encoding="utf-8")
    metrics_frame.to_csv(metrics_path, index=False)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    plot_optimization_history(study, history_chart_path)
    plot_param_importance(study, importance_chart_path)
    return study_path, metrics_path, report_path, history_chart_path, importance_chart_path


def study_to_json(study: optuna.Study) -> str:
    payload = {
        "study_name": study.study_name,
        "direction": str(study.direction),
        "best_value": study.best_value if study.best_trial is not None else None,
        "best_params": study.best_params,
        "trials": [
            {
                "number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "state": str(trial.state),
            }
            for trial in study.trials
        ],
    }
    return json.dumps(payload, indent=2, default=str)


def plot_optimization_history(study: optuna.Study, output_path: Path) -> Path:
    trials = [trial for trial in study.trials if trial.value is not None]
    if not trials:
        output_path.write_text("No trials available.", encoding="utf-8")
        return output_path

    values = [float(trial.value) for trial in trials]
    best_so_far = []
    running_best = None
    for value in values:
        running_best = value if running_best is None else min(running_best, value) if study.direction.name == "MINIMIZE" else max(running_best, value)
        best_so_far.append(running_best)

    plt.figure(figsize=(12, 6))
    plt.plot([trial.number for trial in trials], values, marker="o", linewidth=1.5, label="Trial value")
    plt.plot([trial.number for trial in trials], best_so_far, linewidth=2.0, label="Best so far")
    plt.title(f"Optuna History - {study.study_name}")
    plt.xlabel("Trial")
    plt.ylabel("Objective value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def plot_param_importance(study: optuna.Study, output_path: Path) -> Path:
    try:
        importances = optuna.importance.get_param_importances(study)
    except Exception:
        importances = {}

    plt.figure(figsize=(12, 6))
    if importances:
        names = list(importances.keys())[::-1]
        values = list(importances.values())[::-1]
        plt.barh(names, values, color="#1f77b4")
    else:
        plt.text(0.5, 0.5, "No importance data available", ha="center", va="center")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
    plt.title(f"Parameter Importance - {study.study_name}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def _log_trial_run(run_name: str, params: dict[str, Any], metrics: dict[str, float], artifact_frame: pd.DataFrame | None = None) -> None:
    with start_mlflow_run(run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        if artifact_frame is not None:
            log_dataframe_artifact(artifact_frame, f"{run_name}_trials.csv")


def optimize_churn_model(n_trials: int = 8) -> OptimizationResult:
    ensure_output_dirs()
    ensure_optuna_dirs()

    customer_base = load_customer_base()
    churn_frame = build_churn_dataset(customer_base)
    X, y = build_feature_matrix(churn_frame)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns)
    X_valid_imp = pd.DataFrame(imputer.transform(X_valid), columns=X.columns)
    dtrain = xgb.DMatrix(X_train_imp, label=y_train)
    dvalid = xgb.DMatrix(X_valid_imp, label=y_valid)

    trial_rows: list[dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 450),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 8.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 6.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 3.0),
        }
        scale_pos_weight = max(int((1 - y_train).sum()), 1) / max(int(y_train.sum()), 1)
        train_params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "scale_pos_weight": scale_pos_weight,
            **params,
        }
        booster = xgb.train(
            train_params,
            dtrain,
            num_boost_round=params["n_estimators"],
            evals=[(dvalid, "validation")],
            early_stopping_rounds=25,
            verbose_eval=False,
        )
        probabilities = booster.predict(dvalid, iteration_range=(0, booster.best_iteration + 1 if booster.best_iteration is not None else params["n_estimators"]))
        auc = float(roc_auc_score(y_valid, probabilities))
        trial.report(auc, step=0)
        if trial.should_prune():
            raise TrialPruned()

        with start_mlflow_run("day11_optuna_churn_trial") as run:
            mlflow.log_params(params | {"scale_pos_weight": scale_pos_weight})
            mlflow.log_metric("roc_auc", auc)
            mlflow.log_metric("best_iteration", float(getattr(booster, "best_iteration", 0) or 0))

        trial_rows.append({"trial": trial.number, "roc_auc": auc, **params})
        return auc

    study = optuna.create_study(direction="maximize", study_name="retailpulse_churn_optuna")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_model_params = study.best_params | {"objective": "binary:logistic", "eval_metric": "auc", "random_state": 42, "tree_method": "hist"}
    best_scale = max(int((1 - y_train).sum()), 1) / max(int(y_train.sum()), 1)
    best_train_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "scale_pos_weight": best_scale,
        **study.best_params,
    }
    best_booster = xgb.train(
        best_train_params,
        dtrain,
        num_boost_round=study.best_params["n_estimators"],
        evals=[(dvalid, "validation")],
        early_stopping_rounds=25,
        verbose_eval=False,
    )
    full_imputer = SimpleImputer(strategy="median")
    X_full_imp = pd.DataFrame(full_imputer.fit_transform(X), columns=X.columns)
    full_dmatrix = xgb.DMatrix(X_full_imp, label=y)
    optimized_booster = xgb.train(
        {**best_train_params, "scale_pos_weight": best_scale},
        full_dmatrix,
        num_boost_round=best_booster.best_iteration + 1 if best_booster.best_iteration is not None else study.best_params["n_estimators"],
        verbose_eval=False,
    )

    model_path = OPTUNA_MODELS_DIR / "optuna_churn_xgboost_pipeline.joblib"
    joblib.dump({"imputer": full_imputer, "booster": optimized_booster, "best_params": study.best_params, "scale_pos_weight": best_scale, "feature_columns": list(X.columns)}, model_path)

    metrics = {
        "best_roc_auc": float(study.best_value),
        "validation_roc_auc": float(roc_auc_score(y_valid, best_booster.predict(dvalid, iteration_range=(0, best_booster.best_iteration + 1 if best_booster.best_iteration is not None else study.best_params["n_estimators"])))),
    }
    metrics_frame = pd.DataFrame([{"metric": key, "value": value} for key, value in metrics.items()])
    report_lines = [
        "# Day 11 - Optuna Churn Optimization",
        "",
        pd.DataFrame([study.best_params]).to_markdown(index=False),
        "",
        metrics_frame.to_markdown(index=False),
        "",
        "The best XGBoost configuration is the one with the highest validation ROC AUC.",
    ]
    study_path, metrics_path, report_path, history_chart_path, importance_chart_path = save_study_artifacts(study, "churn_optuna", metrics_frame, report_lines)

    setup_mlflow("RetailPulse")
    with start_mlflow_run("day11_optuna_churn_summary") as run:
        mlflow.log_params(study.best_params | {"scale_pos_weight": best_scale})
        mlflow.log_metrics(metrics)
        log_json_artifact({"best_params": study.best_params, "scale_pos_weight": best_scale}, "churn_optuna_best_params.json")
        log_dataframe_artifact(metrics_frame, "churn_optuna_metrics.csv")
        log_text_artifact(report_path.read_text(encoding="utf-8"), "churn_optuna_report.md")
        mlflow.xgboost.log_model(optimized_booster, artifact_path="churn_optuna_booster")
        safe_register_model(f"runs:/{run.info.run_id}/churn_optuna_booster", "RetailPulseChurnOptunaModel")

    return OptimizationResult(
        study_name=study.study_name,
        best_params=study.best_params,
        best_metrics=metrics,
        study_path=study_path,
        metrics_path=metrics_path,
        report_path=report_path,
        history_chart_path=history_chart_path,
        importance_chart_path=importance_chart_path,
        model_path=model_path,
    )


def _prepare_lstm_dataloaders(sequence_length: int, batch_size: int):
    sales = load_lstm_daily_sales()
    series = sales["total_price"].astype(float).to_numpy()
    train_cutoff = int(len(series) * 0.7)
    scaler = MinMaxScaler()
    scaler.fit(series[:train_cutoff].reshape(-1, 1))
    scaled_series = scaler.transform(series.reshape(-1, 1)).reshape(-1)
    sequences, targets, target_dates = build_sequence_arrays(scaled_series, sales["date"], sequence_length)
    (train_sequences, train_targets, train_dates), (val_sequences, val_targets, val_dates), (test_sequences, test_targets, test_dates) = split_sequences(
        sequences, targets, target_dates, sequence_length
    )
    return {
        "sales": sales,
        "scaler": scaler,
        "train": (train_sequences, train_targets, train_dates),
        "val": (val_sequences, val_targets, val_dates),
        "test": (test_sequences, test_targets, test_dates),
        "sequence_length": sequence_length,
    }


def optimize_lstm_model(n_trials: int = 6) -> OptimizationResult:
    ensure_output_dirs()
    ensure_optuna_dirs()
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader, Dataset

    class SequenceDataset(Dataset):
        def __init__(self, sequences, targets):
            self.sequences = torch.tensor(sequences, dtype=torch.float32)
            self.targets = torch.tensor(targets, dtype=torch.float32)

        def __len__(self):
            return len(self.targets)

        def __getitem__(self, idx):
            return self.sequences[idx], self.targets[idx]

    def build_loaders(sequence_length: int, batch_size: int):
        dataloaders = _prepare_lstm_dataloaders(sequence_length=sequence_length, batch_size=batch_size)
        train_sequences, train_targets, _ = dataloaders["train"]
        val_sequences, val_targets, _ = dataloaders["val"]
        test_sequences, test_targets, _ = dataloaders["test"]
        train_loader = DataLoader(SequenceDataset(train_sequences, train_targets), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(SequenceDataset(val_sequences, val_targets), batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(SequenceDataset(test_sequences, test_targets), batch_size=batch_size, shuffle=False)
        return dataloaders, train_loader, val_loader, test_loader

    trial_rows: list[dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        sequence_length = trial.suggest_int("sequence_length", 14, 45)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        hidden_size = trial.suggest_int("hidden_size", 32, 96)
        num_layers = trial.suggest_int("num_layers", 1, 3)
        dropout = trial.suggest_float("dropout", 0.0, 0.4)
        learning_rate = trial.suggest_float("learning_rate", 5e-4, 5e-3, log=True)
        max_epochs = trial.suggest_int("max_epochs", 15, 30)

        dataloaders, train_loader, val_loader, _ = build_loaders(sequence_length=sequence_length, batch_size=batch_size)
        model = RetailDemandLSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            learning_rate=learning_rate,
        )
        pruning_callback = LightningPruningCallback(trial, monitor="val_loss")
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            callbacks=[pl.callbacks.EarlyStopping(monitor="val_loss", patience=4, mode="min"), pruning_callback],
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            accelerator="cpu",
            deterministic=True,
        )
        trainer.fit(model, train_loader, val_loader)
        val_loss = float(trainer.callback_metrics["val_loss"].detach().cpu().item())
        trial.report(val_loss, step=trainer.current_epoch)
        if trial.should_prune():
            raise TrialPruned()

        with start_mlflow_run("day11_optuna_lstm_trial") as run:
            mlflow.log_params(
                {
                    "sequence_length": sequence_length,
                    "batch_size": batch_size,
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "dropout": dropout,
                    "learning_rate": learning_rate,
                    "max_epochs": max_epochs,
                }
            )
            mlflow.log_metric("val_loss", val_loss)

        trial_rows.append({"trial": trial.number, "val_loss": val_loss, "sequence_length": sequence_length, "batch_size": batch_size, "hidden_size": hidden_size, "num_layers": num_layers, "dropout": dropout, "learning_rate": learning_rate, "max_epochs": max_epochs})
        return val_loss

    study = optuna.create_study(direction="minimize", study_name="retailpulse_lstm_optuna")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    dataloaders, train_loader, val_loader, test_loader = build_loaders(sequence_length=int(best["sequence_length"]), batch_size=int(best["batch_size"]))
    model = RetailDemandLSTM(
        input_size=1,
        hidden_size=int(best["hidden_size"]),
        num_layers=int(best["num_layers"]),
        dropout=float(best["dropout"]),
        learning_rate=float(best["learning_rate"]),
    )
    checkpoint_dir = OPTUNA_MODELS_DIR
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=str(checkpoint_dir), filename="optuna_lstm_best", monitor="val_loss", mode="min", save_top_k=1)
    trainer = pl.Trainer(
        max_epochs=int(best["max_epochs"]),
        callbacks=[checkpoint_callback, pl.callbacks.EarlyStopping(monitor="val_loss", patience=4, mode="min")],
        logger=False,
        enable_checkpointing=True,
        enable_progress_bar=False,
        accelerator="cpu",
        deterministic=True,
    )
    trainer.fit(model, train_loader, val_loader)
    best_model = RetailDemandLSTM.load_from_checkpoint(checkpoint_callback.best_model_path)
    best_model.eval()

    test_sequences, test_targets, test_dates = dataloaders["test"]
    predictions_scaled = []
    with torch.no_grad():
        for batch_sequences, _ in test_loader:
            predictions_scaled.extend(best_model(batch_sequences).cpu().numpy().tolist())
    actual = inverse_transform(np.array(test_targets), dataloaders["scaler"])
    predicted = inverse_transform(np.array(predictions_scaled), dataloaders["scaler"])
    metrics = lstm_compute_metrics(actual, predicted)
    prediction_frame = pd.DataFrame({"date": pd.to_datetime(test_dates), "actual": actual, "predicted": predicted})
    prediction_frame["error"] = prediction_frame["actual"] - prediction_frame["predicted"]

    model_path = OPTUNA_MODELS_DIR / "optuna_lstm_model_state_dict.pt"
    torch.save(best_model.state_dict(), model_path)
    config_path = OPTUNA_MODELS_DIR / "optuna_lstm_config.json"
    config_path.write_text(json.dumps(best, indent=2), encoding="utf-8")
    scaler_path = OPTUNA_MODELS_DIR / "optuna_lstm_scaler.pkl"
    with open(scaler_path, "wb") as handle:
        pickle.dump(dataloaders["scaler"], handle)

    metrics_frame = pd.DataFrame([{"metric": key, "value": value} for key, value in metrics.items()])
    report_lines = [
        "# Day 11 - Optuna LSTM Optimization",
        "",
        pd.DataFrame([best]).to_markdown(index=False),
        "",
        metrics_frame.to_markdown(index=False),
        "",
        "The optimized LSTM is chosen on the lowest validation loss before evaluation on the held-out test window.",
    ]
    study_path, metrics_path, report_path, history_chart_path, importance_chart_path = save_study_artifacts(study, "lstm_optuna", metrics_frame, report_lines)
    prediction_frame.to_csv(OPTUNA_DIR / "lstm_optuna_predictions.csv", index=False)

    setup_mlflow("RetailPulse")
    with start_mlflow_run("day11_optuna_lstm_summary") as run:
        mlflow.log_params(best)
        mlflow.log_metrics(metrics)
        log_json_artifact({"best_params": best}, "lstm_optuna_best_params.json")
        log_dataframe_artifact(metrics_frame, "lstm_optuna_metrics.csv")
        log_dataframe_artifact(prediction_frame, "lstm_optuna_predictions.csv")
        log_text_artifact(report_path.read_text(encoding="utf-8"), "lstm_optuna_report.md")
        mlflow.pytorch.log_model(best_model, artifact_path="lstm_optuna_model")
        safe_register_model(f"runs:/{run.info.run_id}/lstm_optuna_model", "RetailPulseLSTMOptunaModel")

    return OptimizationResult(
        study_name=study.study_name,
        best_params=best,
        best_metrics=metrics,
        study_path=study_path,
        metrics_path=metrics_path,
        report_path=report_path,
        history_chart_path=history_chart_path,
        importance_chart_path=importance_chart_path,
        model_path=model_path,
    )


def optimize_ensemble_weights(n_trials: int = 20) -> OptimizationResult:
    ensure_output_dirs()
    ensure_optuna_dirs()
    evaluation, _, _, _ = build_evaluation_frames()

    trial_rows: list[dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        prophet_weight = trial.suggest_float("prophet_weight", 0.1, 0.9)
        lstm_weight = 1.0 - prophet_weight
        ensemble_prediction = prophet_weight * evaluation["prophet_yhat"] + lstm_weight * evaluation["lstm_yhat"]
        rmse = float(np.sqrt(mean_squared_error(evaluation["total_price"], ensemble_prediction)))
        trial.report(rmse, step=0)
        if trial.should_prune():
            raise TrialPruned()

        with start_mlflow_run("day11_optuna_ensemble_trial") as run:
            mlflow.log_params({"prophet_weight": prophet_weight, "lstm_weight": lstm_weight})
            mlflow.log_metric("rmse", rmse)

        trial_rows.append({"trial": trial.number, "rmse": rmse, "prophet_weight": prophet_weight, "lstm_weight": lstm_weight})
        return rmse

    study = optuna.create_study(direction="minimize", study_name="retailpulse_ensemble_optuna")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_weights = {"prophet": float(study.best_params["prophet_weight"]), "lstm": float(1.0 - study.best_params["prophet_weight"])}
    optimized_ensemble = run_hybrid_forecasting_ensemble(configured_weights=best_weights)
    metrics = pd.read_csv(optimized_ensemble.metrics_path)
    best_metrics = {
        "ensemble_rmse": float(metrics.loc[metrics["model"] == "Ensemble", "RMSE"].iloc[0]),
        "ensemble_mae": float(metrics.loc[metrics["model"] == "Ensemble", "MAE"].iloc[0]),
        "ensemble_mape": float(metrics.loc[metrics["model"] == "Ensemble", "MAPE"].iloc[0]),
    }

    metrics_frame = pd.DataFrame([{"metric": key, "value": value} for key, value in best_metrics.items()])
    report_lines = [
        "# Day 11 - Optuna Ensemble Weight Optimization",
        "",
        pd.DataFrame([best_weights]).to_markdown(index=False),
        "",
        metrics_frame.to_markdown(index=False),
        "",
        "The best ensemble is selected by minimizing validation RMSE on the forecast comparison window.",
    ]
    study_path, metrics_path, report_path, history_chart_path, importance_chart_path = save_study_artifacts(study, "ensemble_optuna", metrics_frame, report_lines)

    setup_mlflow("RetailPulse")
    with start_mlflow_run("day11_optuna_ensemble_summary") as run:
        mlflow.log_params(best_weights)
        mlflow.log_metrics(best_metrics)
        log_json_artifact({"best_weights": best_weights}, "ensemble_optuna_best_weights.json")
        log_dataframe_artifact(metrics_frame, "ensemble_optuna_metrics.csv")
        log_text_artifact(report_path.read_text(encoding="utf-8"), "ensemble_optuna_report.md")
        safe_register_model(f"runs:/{run.info.run_id}/ensemble_optuna_model", "RetailPulseEnsembleOptunaModel")

    return OptimizationResult(
        study_name=study.study_name,
        best_params=best_weights,
        best_metrics=best_metrics,
        study_path=study_path,
        metrics_path=metrics_path,
        report_path=report_path,
        history_chart_path=history_chart_path,
        importance_chart_path=importance_chart_path,
        model_path=optimized_ensemble.model_path,
    )


def run_optuna_optimization_suite(churn_trials: int = 8, lstm_trials: int = 6, ensemble_trials: int = 20) -> dict[str, OptimizationResult]:
    return {
        "churn": optimize_churn_model(n_trials=churn_trials),
        "lstm": optimize_lstm_model(n_trials=lstm_trials),
        "ensemble": optimize_ensemble_weights(n_trials=ensemble_trials),
    }
