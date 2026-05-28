from __future__ import annotations

from dataclasses import dataclass

import altair as alt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
# Heavy imports moved to functions to avoid startup blocking


@dataclass(frozen=True)
class WhatIfScenario:
    sales_growth_pct: float
    seasonality_multiplier: float
    inventory_capacity_multiplier: float
    demand_shift_pct: float


def forecast_comparison_figure(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if frame.empty:
        return fig
    fig.add_trace(go.Scatter(x=frame["date"], y=frame["total_price"], name="Actual", line=dict(color="#1f77b4", width=2)))
    fig.add_trace(go.Scatter(x=frame["date"], y=frame["prophet_yhat"], name="Prophet", line=dict(color="#ff7f0e", width=2)))
    fig.add_trace(go.Scatter(x=frame["date"], y=frame["lstm_yhat"], name="LSTM", line=dict(color="#2ca02c", width=2)))
    fig.add_trace(go.Scatter(x=frame["date"], y=frame["ensemble_yhat"], name="Ensemble", line=dict(color="#d62728", width=3)))
    fig.update_layout(template="plotly_dark", height=480, margin=dict(l=20, r=20, t=40, b=20), legend=dict(orientation="h"))
    return fig


def forecast_band_figure(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if frame.empty:
        return fig
    fig.add_trace(go.Scatter(x=frame["date"], y=frame["forecast_upper_quantity"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=frame["date"], y=frame["forecast_lower_quantity"], fill="tonexty", line=dict(width=0), name="Prediction Band", fillcolor="rgba(239,68,68,0.18)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=frame["date"], y=frame["ensemble_quantity_forecast"], name="Forecast Demand", line=dict(color="#38bdf8", width=3)))
    fig.update_layout(template="plotly_dark", height=420, margin=dict(l=20, r=20, t=30, b=20))
    return fig


def product_level_forecast_figure(frame: pd.DataFrame) -> go.Figure:
    if frame.empty:
        return go.Figure()
    ordered = frame.sort_values("ensemble_quantity_forecast", ascending=False).head(10)
    fig = px.bar(ordered, x="date", y="ensemble_quantity_forecast", color="ensemble_quantity_forecast", template="plotly_dark", title="Top Forecast Horizon")
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def what_if_simulation(frame: pd.DataFrame, scenario: WhatIfScenario, baseline_col: str = "ensemble_quantity_forecast") -> pd.DataFrame:
    if frame.empty:
        return frame
    output = frame.copy()
    growth_factor = 1 + scenario.sales_growth_pct / 100.0
    demand_factor = 1 + scenario.demand_shift_pct / 100.0
    output["what_if_quantity"] = output[baseline_col] * growth_factor * scenario.seasonality_multiplier * demand_factor
    output["inventory_pressure"] = output["what_if_quantity"] / max(scenario.inventory_capacity_multiplier, 1e-6)
    output["scenario_label"] = f"Growth {scenario.sales_growth_pct:.0f}% / Seasonality x{scenario.seasonality_multiplier:.2f}"
    return output


def what_if_chart(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if frame.empty:
        return fig
    fig.add_trace(go.Scatter(x=frame["date"], y=frame["ensemble_quantity_forecast"], name="Baseline", line=dict(color="#94a3b8", width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=frame["date"], y=frame["what_if_quantity"], name="What-if Scenario", line=dict(color="#f59e0b", width=3)))
    fig.add_trace(go.Bar(x=frame["date"], y=frame["inventory_pressure"], name="Inventory Pressure", opacity=0.25, yaxis="y2"))
    fig.update_layout(
        template="plotly_dark",
        height=450,
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis2=dict(overlaying="y", side="right", title="Pressure"),
        legend=dict(orientation="h"),
    )
    return fig


def cluster_distribution_chart(frame: pd.DataFrame) -> alt.Chart:
    if frame.empty:
        return alt.Chart(pd.DataFrame({"label": [], "count": []})).mark_bar()
    data = frame.copy()
    if "cluster" not in data.columns:
        for candidate in ["kmeans_cluster", "dbscan_cluster", "Cluster"]:
            if candidate in data.columns:
                data = data.rename(columns={candidate: "cluster"})
                break
    if "cluster" not in data.columns:
        data["cluster"] = "Unknown"
    counts = data.groupby("cluster", as_index=False).size().rename(columns={"size": "count"})
    return alt.Chart(counts).mark_bar(color="#38bdf8").encode(x=alt.X("cluster:N", sort="-y"), y="count:Q", tooltip=["cluster", "count"])


def _embedding_source(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    candidate_columns = [
        "Recency",
        "Frequency",
        "Monetary",
        "R_Score",
        "F_Score",
        "M_Score",
        "RFM_Score",
        "RFM_Total",
        "pca_1",
        "pca_2",
    ]
    available = [column for column in candidate_columns if column in frame.columns]
    if not available:
        numeric = frame.select_dtypes(include=[np.number]).columns.tolist()
        available = numeric[:8]
    if not available:
        return frame.copy()
    source = frame[available + [column for column in ["persona", "Country", "kmeans_cluster", "dbscan_cluster"] if column in frame.columns]].copy()
    for column in available:
        source[column] = pd.to_numeric(source[column], errors="coerce")
    source = source.dropna(subset=available)
    return source


def segmentation_embedding_frame(frame: pd.DataFrame, method: str = "pca", sample_size: int = 1200) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    source = _embedding_source(frame)
    if source.empty:
        return source

    numeric_columns = source.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_columns:
        return source

    if len(source) > sample_size:
        source = source.sample(sample_size, random_state=42)

    try:
        if method.lower() == "tsne" and len(source) >= 5:
            from sklearn.preprocessing import StandardScaler
            from sklearn.manifold import TSNE

            features = source[numeric_columns].fillna(0.0).to_numpy()
            scaled = StandardScaler().fit_transform(features)
            perplexity = min(30, max(5, len(source) // 20))
            reducer = TSNE(n_components=2, perplexity=perplexity, learning_rate="auto", init="pca", random_state=42)
            embedding = reducer.fit_transform(scaled)
            source = source.copy()
            source["embedding_x"] = embedding[:, 0]
            source["embedding_y"] = embedding[:, 1]
        else:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler

            components = 2 if len(numeric_columns) >= 2 else 1
            reducer = PCA(n_components=components, random_state=42)
            scaled = StandardScaler().fit_transform(source[numeric_columns].fillna(0.0).to_numpy())
            embedding = reducer.fit_transform(scaled)
            source = source.copy()
            source["embedding_x"] = embedding[:, 0]
            source["embedding_y"] = embedding[:, 1] if embedding.shape[1] > 1 else 0.0
    except Exception:
        # If embedding fails, return source with zeros so page doesn't crash
        source = source.copy()
        source["embedding_x"] = 0.0
        source["embedding_y"] = 0.0

    return source


def segmentation_embedding_chart(frame: pd.DataFrame, method: str = "pca", color_col: str = "persona") -> go.Figure:
    if frame.empty:
        return go.Figure()
    embedding = segmentation_embedding_frame(frame, method=method)
    if embedding.empty:
        return go.Figure()
    color_field = color_col if color_col in embedding.columns else ("Country" if "Country" in embedding.columns else None)
    if color_field is None:
        color_field = "embedding_x"
    fig = px.scatter(
        embedding,
        x="embedding_x",
        y="embedding_y",
        color=color_field,
        hover_data=[column for column in ["CustomerID", "Country", "persona", "kmeans_cluster", "dbscan_cluster", "Monetary", "Recency", "Frequency"] if column in embedding.columns],
        template="plotly_dark",
        title=f"{method.upper()} customer embedding",
    )
    fig.update_layout(height=520, margin=dict(l=20, r=20, t=50, b=20), legend=dict(orientation="h"))
    return fig


def segment_profile_chart(frame: pd.DataFrame, label_column: str = "persona") -> go.Figure:
    if frame.empty or label_column not in frame.columns:
        return go.Figure()
    counts = frame[label_column].astype(str).value_counts().reset_index()
    counts.columns = [label_column, "count"]
    fig = px.bar(counts, x=label_column, y="count", color="count", template="plotly_dark", title=f"{label_column.title()} distribution")
    fig.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
    return fig


def churn_risk_chart(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if frame.empty:
        return fig
    if "risk_band" not in frame.columns:
        return fig
    counts = frame["risk_band"].astype(str).value_counts().reset_index()
    counts.columns = ["risk_band", "count"]
    fig = px.pie(counts, names="risk_band", values="count", hole=0.55, template="plotly_dark", title="Churn Risk Distribution")
    fig.update_layout(height=420)
    return fig


def churn_probability_profile_chart(frame: pd.DataFrame) -> go.Figure:
    if frame.empty or "churn_probability" not in frame.columns:
        return go.Figure()
    ordered = frame.sort_values("churn_probability", ascending=False).reset_index(drop=True)
    ordered["rank"] = np.arange(1, len(ordered) + 1)
    ordered["rolling_mean"] = ordered["churn_probability"].rolling(window=max(5, len(ordered) // 20), min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ordered["rank"], y=ordered["churn_probability"], name="Customer risk", line=dict(color="#f97316", width=2)))
    fig.add_trace(go.Scatter(x=ordered["rank"], y=ordered["rolling_mean"], name="Rolling average", line=dict(color="#38bdf8", width=3)))
    fig.update_layout(template="plotly_dark", height=420, margin=dict(l=20, r=20, t=40, b=20), xaxis_title="Risk rank", yaxis_title="Churn probability")
    return fig


def churn_feature_profile_chart(customer_row: pd.Series, selected_features: list[str]) -> go.Figure:
    if customer_row.empty or not selected_features:
        return go.Figure()
    values = pd.Series({feature: float(customer_row.get(feature, 0.0)) for feature in selected_features})
    ordered = values.sort_values(ascending=False).head(10).reset_index()
    ordered.columns = ["feature", "value"]
    fig = px.bar(ordered, x="feature", y="value", template="plotly_dark", title="Selected customer feature profile")
    fig.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
    fig.update_xaxes(title_text="Feature")
    fig.update_yaxes(title_text="Value")
    return fig


def inventory_risk_chart(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if frame.empty:
        return fig
    ordered = frame.sort_values("risk_flag").head(12)
    fig.add_trace(go.Bar(x=ordered["StockCode"], y=ordered["current_stock"], name="Current Stock", marker_color="#1f77b4"))
    fig.add_trace(go.Bar(x=ordered["StockCode"], y=ordered["reorder_point"], name="Reorder Point", marker_color="#f97316"))
    fig.update_layout(template="plotly_dark", barmode="group", height=430, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def inventory_heatmap_chart(frame: pd.DataFrame) -> go.Figure:
    if frame.empty:
        return go.Figure()
    ordered = frame.sort_values("risk_flag").head(12).copy()
    matrix_columns = [column for column in ["current_stock", "safety_stock", "reorder_point", "target_stock_level", "days_of_cover", "recommended_order_qty"] if column in ordered.columns]
    if not matrix_columns:
        return go.Figure()
    heatmap_data = ordered.set_index("StockCode")[matrix_columns].T
    fig = px.imshow(heatmap_data, aspect="auto", color_continuous_scale="Blues", template="plotly_dark", title="Inventory health heatmap")
    fig.update_layout(height=430, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def reorder_timeline_chart(frame: pd.DataFrame) -> go.Figure:
    if frame.empty:
        return go.Figure()
    ordered = frame.sort_values(["days_of_cover", "recommended_order_qty"], ascending=[True, False]).head(15)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=ordered["StockCode"], y=ordered["recommended_order_qty"], name="Recommended order qty", marker_color="#38bdf8"))
    fig.add_trace(go.Scatter(x=ordered["StockCode"], y=ordered["days_of_cover"], name="Days of cover", line=dict(color="#f59e0b", width=3), yaxis="y2"))
    fig.update_layout(
        template="plotly_dark",
        height=450,
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis2=dict(overlaying="y", side="right", title="Days of cover"),
        legend=dict(orientation="h"),
    )
    return fig


def drift_summary_chart(metrics: dict[str, float]) -> go.Figure:
    keys = ["data_drift_score", "prediction_drift_score", "model_reliability_score", "forecasting_degradation_risk"]
    values = [metrics.get(key, 0.0) for key in keys]
    fig = go.Figure(go.Bar(x=keys, y=values, marker_color=["#ef4444", "#f59e0b", "#10b981", "#3b82f6"]))
    fig.update_layout(template="plotly_dark", height=360, margin=dict(l=20, r=20, t=40, b=20), yaxis_title="Score")
    return fig


def drift_breakdown_chart(drift_summary: dict[str, float]) -> go.Figure:
    if not drift_summary:
        return go.Figure()
    keys = list(drift_summary.keys())
    values = [drift_summary[key] for key in keys]
    fig = go.Figure(go.Bar(x=keys, y=values, marker_color="#8b5cf6"))
    fig.update_layout(template="plotly_dark", height=400, margin=dict(l=20, r=20, t=40, b=20), xaxis_title="Signal", yaxis_title="Shift")
    return fig


def health_gauge(value: float, title: str, threshold_warning: float = 0.4, threshold_bad: float = 0.6) -> go.Figure:
    color = "#10b981" if value < threshold_warning else "#f59e0b" if value < threshold_bad else "#ef4444"
    fig = go.Figure(go.Indicator(mode="gauge+number", value=value, title={"text": title}, gauge={"axis": {"range": [0, 1]}, "bar": {"color": color}}))
    fig.update_layout(template="plotly_dark", height=280, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def mlflow_metrics_chart(frame: pd.DataFrame) -> go.Figure:
    if frame.empty:
        return go.Figure()
    if "metric" in frame.columns and "value" in frame.columns:
        data = frame.copy()
    else:
        data = frame.melt(var_name="metric", value_name="value")
    fig = px.bar(data, x="metric", y="value", color="metric", template="plotly_dark", title="ML Metrics")
    fig.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
    return fig


def top_shap_features_chart(frame: pd.DataFrame) -> go.Figure:
    if frame.empty or not {"feature", "mean_abs_shap"}.issubset(frame.columns):
        return go.Figure()
    ordered = frame.sort_values("mean_abs_shap", ascending=True).tail(10)
    fig = px.bar(ordered, x="mean_abs_shap", y="feature", orientation="h", template="plotly_dark", title="Top SHAP Drivers")
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
    return fig


def customer_explanation_table(customer_row: pd.Series, shap_ranking: pd.DataFrame) -> pd.DataFrame:
    if customer_row.empty or shap_ranking.empty or "feature" not in shap_ranking.columns:
        return pd.DataFrame()
    rows = []
    for _, shap_row in shap_ranking.head(10).iterrows():
        feature = str(shap_row["feature"])
        if feature not in customer_row.index:
            continue
        value = customer_row[feature]
        rows.append(
            {
                "feature": feature,
                "value": value,
                "mean_abs_shap": float(shap_row.get("mean_abs_shap", 0.0)),
                "impact_score": float(abs(value) if pd.notna(value) else 0.0) * float(shap_row.get("mean_abs_shap", 0.0)),
            }
        )
    explanation = pd.DataFrame(rows)
    if not explanation.empty:
        explanation = explanation.sort_values("impact_score", ascending=False).reset_index(drop=True)
    return explanation
