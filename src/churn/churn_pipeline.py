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
import shap
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from retailpulse_mlflow_utils import log_dataframe_artifact, log_json_artifact, log_text_artifact, safe_register_model, setup_mlflow, start_mlflow_run

ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "processed"
MODELS_DIR = PROCESSED_DIR / "models"
FIGURES_DIR = PROCESSED_DIR / "figures"


@dataclass
class ChurnRunResult:
    model_path: Path
    predictions_path: Path
    report_path: Path
    shap_summary_path: Path
    shap_waterfall_path: Path
    confusion_matrix_path: Path
    metrics_path: Path
    metrics: dict[str, float]


def load_customer_base() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "customer_base_features.csv", parse_dates=["first_purchase", "last_purchase"])


def build_churn_dataset(customer_base: pd.DataFrame, churn_threshold_days: int = 90) -> pd.DataFrame:
    frame = customer_base.copy()
    frame["inactivity_duration"] = frame["recency_days"].astype(float)
    frame["avg_purchase_interval"] = frame["purchase_span_days"] / frame["frequency"].clip(lower=1)
    frame["order_consistency"] = frame["frequency"] / frame["tenure_days"].clip(lower=1)
    frame["customer_lifetime_value"] = frame["total_revenue"] * (1.0 + frame["tenure_days"].clip(lower=0) / 365.0)
    frame["purchase_velocity"] = frame["frequency"] / frame["days_since_first_purchase"].clip(lower=1)
    frame["monetary_per_day"] = frame["total_revenue"] / frame["days_since_first_purchase"].clip(lower=1)
    frame["basket_value_stability"] = frame["avg_unit_price"] / frame["avg_basket_size"].clip(lower=1)
    frame["churn_label"] = (frame["recency_days"] >= churn_threshold_days).astype(int)
    frame["churn_status"] = np.select(
        [frame["recency_days"] >= 180, frame["recency_days"] >= churn_threshold_days],
        ["Lost", "At Risk"],
        default="Active",
    )
    return frame


def build_feature_matrix(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    feature_columns = [
        "recency_days",
        "frequency",
        "total_revenue",
        "avg_unit_price",
        "avg_basket_size",
        "orders_per_day",
        "revenue_per_order",
        "units_per_order",
        "tenure_days",
        "purchase_span_days",
        "inactivity_duration",
        "avg_purchase_interval",
        "order_consistency",
        "customer_lifetime_value",
        "purchase_velocity",
        "monetary_per_day",
        "basket_value_stability",
    ]
    X = frame[feature_columns].replace([np.inf, -np.inf], np.nan)
    y = frame["churn_label"].astype(int)
    return X, y


def train_churn_model(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    positive_count = max(int(y_train.sum()), 1)
    negative_count = max(int((1 - y_train).sum()), 1)
    scale_pos_weight = negative_count / positive_count
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_lambda=1.0,
                    min_child_weight=1,
                    scale_pos_weight=scale_pos_weight,
                    objective="binary:logistic",
                    eval_metric="auc",
                    random_state=42,
                    tree_method="hist",
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[pd.DataFrame, dict[str, float], np.ndarray]:
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
    }
    result_frame = X_test.copy()
    result_frame["actual_churn"] = y_test.values
    result_frame["churn_probability"] = probabilities
    result_frame["predicted_churn"] = predictions
    result_frame["risk_band"] = pd.cut(
        result_frame["churn_probability"],
        bins=[-0.01, 0.33, 0.66, 1.0],
        labels=["Low", "Medium", "High"],
    )
    return result_frame, metrics, confusion_matrix(y_test, predictions)


def save_confusion_matrix(matrix: np.ndarray, output_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_title("Churn Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for (row, col), value in np.ndenumerate(matrix):
        ax.text(col, row, int(value), ha="center", va="center", color="black", fontsize=12)
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def save_shap_artifacts(model: Pipeline, X_test: pd.DataFrame, output_prefix: str = "churn_shap") -> tuple[Path, Path, pd.DataFrame]:
    imputed = pd.DataFrame(model.named_steps["imputer"].transform(X_test), columns=X_test.columns, index=X_test.index)
    classifier = model.named_steps["model"]
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(imputed)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    summary_path = FIGURES_DIR / f"{output_prefix}_summary.png"
    waterfall_path = FIGURES_DIR / f"{output_prefix}_waterfall.png"

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, imputed, show=False, plot_type="bar")
    plt.tight_layout()
    plt.savefig(summary_path, dpi=160, bbox_inches="tight")
    plt.close()

    target_index = int(np.argmax(classifier.predict_proba(imputed)[:, 1]))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[target_index],
            base_values=explainer.expected_value if not isinstance(explainer.expected_value, list) else explainer.expected_value[1],
            data=imputed.iloc[target_index],
            feature_names=list(imputed.columns),
        ),
        show=False,
    )
    plt.tight_layout()
    plt.savefig(waterfall_path, dpi=160, bbox_inches="tight")
    plt.close()

    shap_frame = pd.DataFrame({
        "feature": imputed.columns,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    return summary_path, waterfall_path, shap_frame


def build_report(metrics: dict[str, float], churn_results: pd.DataFrame, shap_frame: pd.DataFrame, report_path: Path, threshold_days: int) -> None:
    high_risk = churn_results.sort_values("churn_probability", ascending=False).head(10).copy()
    lines = [
        "# RetailPulse Churn Prediction Report",
        "",
        f"## Churn Logic",
        "",
        f"Customers are labeled as churned when inactivity reaches at least {threshold_days} days.",
        "",
        "## Evaluation Metrics",
        "",
        pd.DataFrame([metrics]).to_markdown(index=False),
        "",
        "## Top Risk Customers",
        "",
        high_risk[["recency_days", "frequency", "total_revenue", "churn_probability", "risk_band"]].to_markdown(index=False),
        "",
        "## SHAP Feature Ranking",
        "",
        shap_frame.head(12).to_markdown(index=False),
        "",
        "## Business Interpretation",
        "",
        "- High-risk customers are concentrated where recency is high and purchase frequency is low.",
        "- These customers should be prioritized for retention campaigns, win-back offers, and service outreach.",
        "- Recency-driven churn is the strongest signal in this dataset, so operational contact timing matters.",
        "- The SHAP ranking can be used to explain individual retention decisions to the business team.",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_churn_prediction_pipeline(churn_threshold_days: int = 90) -> ChurnRunResult:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    customer_base = load_customer_base()
    churn_frame = build_churn_dataset(customer_base, churn_threshold_days=churn_threshold_days)
    X, y = build_feature_matrix(churn_frame)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

    model = train_churn_model(X_train, y_train)
    evaluation_frame, metrics, confusion = evaluate_model(model, X_test, y_test)
    shap_summary_path, shap_waterfall_path, shap_frame = save_shap_artifacts(model, X_test)
    confusion_matrix_path = save_confusion_matrix(confusion, FIGURES_DIR / "churn_confusion_matrix.png")

    predictions_path = PROCESSED_DIR / "customer_churn_predictions.csv"
    metrics_path = PROCESSED_DIR / "customer_churn_metrics.csv"
    report_path = PROCESSED_DIR / "churn_prediction_report.md"
    model_path = MODELS_DIR / "churn_xgboost_pipeline.joblib"

    evaluation_frame.to_csv(predictions_path, index=False)
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump({"model": model, "threshold_days": churn_threshold_days, "feature_columns": list(X.columns)}, model_path)
    build_report(metrics, evaluation_frame, shap_frame, report_path, churn_threshold_days)

    setup_mlflow("RetailPulse")
    with start_mlflow_run("day9_churn_prediction") as run:
        mlflow.log_params({"churn_threshold_days": churn_threshold_days, "model_type": "xgboost"})
        mlflow.log_metrics(metrics)
        log_dataframe_artifact(evaluation_frame, "customer_churn_predictions.csv")
        log_dataframe_artifact(pd.DataFrame([metrics]), "customer_churn_metrics.csv")
        log_dataframe_artifact(shap_frame, "customer_churn_shap_ranking.csv")
        log_text_artifact(report_path.read_text(encoding="utf-8"), "churn_prediction_report.md")
        log_json_artifact({"threshold_days": churn_threshold_days, "features": list(X.columns)}, "churn_config.json")
        mlflow.sklearn.log_model(model, artifact_path="churn_model")
        safe_register_model(f"runs:/{run.info.run_id}/churn_model", "RetailPulseChurnClassifier")

    return ChurnRunResult(
        model_path=model_path,
        predictions_path=predictions_path,
        report_path=report_path,
        shap_summary_path=shap_summary_path,
        shap_waterfall_path=shap_waterfall_path,
        confusion_matrix_path=confusion_matrix_path,
        metrics_path=metrics_path,
        metrics=metrics,
    )


if __name__ == "__main__":
    run_churn_prediction_pipeline()
