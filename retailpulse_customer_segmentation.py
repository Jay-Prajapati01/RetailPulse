from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from retailpulse_feature_pipeline import MODELS_DIR, FIGURES_DIR, PROCESSED_DIR, ensure_output_dirs


def _tabulate_available() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


def load_customer_rfm() -> pd.DataFrame:
    file_path = PROCESSED_DIR / "customer_rfm.csv"
    frame = pd.read_csv(file_path)
    numeric_columns = ["Recency", "Frequency", "Monetary"]
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(subset=numeric_columns).reset_index(drop=True)
    return frame


def choose_best_kmeans(features: np.ndarray, k_range: range = range(2, 11)) -> tuple[int, list[dict[str, float]]]:
    results: list[dict[str, float]] = []
    best_k = 2
    best_score = -1.0
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(features)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(features, labels)
        inertia = float(model.inertia_)
        results.append({"k": k, "silhouette": score, "inertia": inertia})
        if score > best_score:
            best_score = score
            best_k = k
    return best_k, results


def choose_best_dbscan(features: np.ndarray) -> tuple[DBSCAN, list[dict[str, float]]]:
    parameter_grid = [
        (0.35, 5),
        (0.45, 5),
        (0.55, 10),
        (0.65, 10),
        (0.75, 10),
        (0.85, 15),
        (1.00, 15),
        (1.15, 15),
    ]
    evaluations: list[dict[str, float]] = []
    best_model = DBSCAN(eps=0.65, min_samples=10)
    best_score = -1.0

    for eps, min_samples in parameter_grid:
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(features)
        non_noise = labels[labels != -1]
        cluster_count = len(set(non_noise))
        if cluster_count < 2 or len(non_noise) < 10:
            evaluations.append({"eps": eps, "min_samples": min_samples, "silhouette": np.nan, "clusters": cluster_count})
            continue
        score = silhouette_score(features[labels != -1], labels[labels != -1])
        evaluations.append({"eps": eps, "min_samples": min_samples, "silhouette": score, "clusters": cluster_count})
        if score > best_score:
            best_score = score
            best_model = model

    return best_model, evaluations


def assign_persona(row: pd.Series, global_frame: pd.DataFrame) -> str:
    recency_q25 = global_frame["Recency"].quantile(0.25)
    recency_q75 = global_frame["Recency"].quantile(0.75)
    frequency_q25 = global_frame["Frequency"].quantile(0.25)
    frequency_q75 = global_frame["Frequency"].quantile(0.75)
    monetary_q25 = global_frame["Monetary"].quantile(0.25)
    monetary_q75 = global_frame["Monetary"].quantile(0.75)

    if row["Recency"] <= recency_q25 and row["Frequency"] >= frequency_q75 and row["Monetary"] >= monetary_q75:
        return "Champions"
    if row["Recency"] <= recency_q25 and row["Frequency"] >= frequency_q25:
        return "Loyal Customers"
    if row["Monetary"] >= monetary_q75 and row["Frequency"] <= frequency_q75:
        return "Big Spenders"
    if row["Frequency"] == 1 and row["Recency"] <= recency_q75:
        return "New Customers"
    if row["Recency"] >= recency_q75 and row["Frequency"] <= frequency_q25:
        return "At Risk"
    if row["Recency"] >= recency_q75 and row["Monetary"] <= monetary_q25:
        return "Lost"
    return "Regular Customers"


def save_cluster_report(clustered: pd.DataFrame, kmeans_score: float, dbscan_score: float, dbscan_model: DBSCAN, report_path: Path) -> None:
    kmeans_profile = clustered.groupby("kmeans_cluster")[["Recency", "Frequency", "Monetary"]].mean().round(2)
    dbscan_profile = clustered.groupby("dbscan_cluster")[["Recency", "Frequency", "Monetary"]].mean().round(2)
    persona_counts = clustered["persona"].value_counts().to_frame(name="customer_count")

    lines = [
        "# RetailPulse Customer Segmentation Report",
        "",
        f"- KMeans silhouette score: {kmeans_score:.4f}",
        f"- DBSCAN silhouette score: {dbscan_score:.4f}" if np.isfinite(dbscan_score) else "- DBSCAN silhouette score: not available",
        f"- DBSCAN eps: {dbscan_model.eps}",
        f"- DBSCAN min_samples: {dbscan_model.min_samples}",
        "",
        "## KMeans Cluster Profile",
        "",
        kmeans_profile.to_markdown() if _tabulate_available() else kmeans_profile.to_csv(),
        "",
        "## DBSCAN Cluster Profile",
        "",
        dbscan_profile.to_markdown() if _tabulate_available() else dbscan_profile.to_csv(),
        "",
        "## Persona Distribution",
        "",
        persona_counts.to_markdown() if _tabulate_available() else persona_counts.to_csv(),
        "",
        "## Business Insights",
        "",
        "- Champions and Loyal Customers should be prioritized for retention, upsell, and loyalty campaigns.",
        "- Big Spenders are ideal for premium bundles, exclusive offers, and white-glove service.",
        "- At Risk and Lost customers should be targeted with win-back journeys and operational root-cause analysis.",
        "- The cluster structure should be revisited after each major promotion cycle to keep the personas current.",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def save_visualizations(clustered: pd.DataFrame) -> None:
    feature_cols = ["Recency", "Frequency", "Monetary"]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(clustered[feature_cols])

    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(scaled)

    tsne_sample = clustered.sample(n=min(len(clustered), 4000), random_state=42).copy()
    tsne_scaled = scaler.fit_transform(tsne_sample[feature_cols])
    perplexity = min(30, max(5, len(tsne_sample) // 50))
    tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=42)
    tsne_coords = tsne.fit_transform(tsne_scaled)

    figures = [
        (pca_coords, clustered["kmeans_cluster"], "kmeans_pca_clusters.png", "KMeans clusters in PCA space"),
        (pca_coords, clustered["dbscan_cluster"], "dbscan_pca_clusters.png", "DBSCAN clusters in PCA space"),
    ]

    for coords, labels, file_name, title in figures:
        plt.figure(figsize=(10, 7))
        scatter = plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=18, alpha=0.8)
        plt.title(title)
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.colorbar(scatter)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / file_name, dpi=160)
        plt.close()

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(tsne_coords[:, 0], tsne_coords[:, 1], c=tsne_sample["kmeans_cluster"], cmap="tab10", s=18, alpha=0.8)
    plt.title("KMeans clusters in t-SNE space")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.colorbar(scatter)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "kmeans_tsne_clusters.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 6))
    plt.plot(range(2, 11), clustered.attrs["kmeans_elbow_inertia"], marker="o")
    plt.title("KMeans Elbow Method")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "kmeans_elbow.png", dpi=160)
    plt.close()


def run_customer_segmentation_pipeline() -> pd.DataFrame:
    ensure_output_dirs()
    customer_rfm = load_customer_rfm()

    feature_cols = ["Recency", "Frequency", "Monetary"]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(customer_rfm[feature_cols])

    best_k, kmeans_grid = choose_best_kmeans(scaled_features)
    elbow_inertia = []
    silhouette_scores = []
    for result in kmeans_grid:
        elbow_inertia.append(result["inertia"])
        silhouette_scores.append(result["silhouette"])

    kmeans_model = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    customer_rfm["kmeans_cluster"] = kmeans_model.fit_predict(scaled_features)

    dbscan_model, dbscan_grid = choose_best_dbscan(scaled_features)
    customer_rfm["dbscan_cluster"] = dbscan_model.fit_predict(scaled_features)

    pca = PCA(n_components=2, random_state=42)
    pca_components = pca.fit_transform(scaled_features)
    customer_rfm["pca_1"] = pca_components[:, 0]
    customer_rfm["pca_2"] = pca_components[:, 1]

    tsne_sample = customer_rfm.sample(n=min(len(customer_rfm), 4000), random_state=42).copy()
    tsne_scaler = StandardScaler()
    tsne_scaled = tsne_scaler.fit_transform(tsne_sample[feature_cols])
    perplexity = min(30, max(5, len(tsne_sample) // 50))
    tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=42)
    tsne_coords = tsne.fit_transform(tsne_scaled)
    tsne_sample["tsne_1"] = tsne_coords[:, 0]
    tsne_sample["tsne_2"] = tsne_coords[:, 1]

    persona_frame = customer_rfm.copy()
    persona_frame["persona"] = persona_frame.apply(lambda row: assign_persona(row, customer_rfm), axis=1)
    cluster_personas = persona_frame.groupby("kmeans_cluster").apply(
        lambda group: assign_persona(group[["Recency", "Frequency", "Monetary"]].mean(), customer_rfm)
    )
    customer_rfm["persona"] = customer_rfm["kmeans_cluster"].map(cluster_personas.to_dict())

    model_path = MODELS_DIR / "customer_scaler.joblib"
    joblib.dump(scaler, model_path)
    joblib.dump(kmeans_model, MODELS_DIR / "customer_kmeans_model.joblib")
    joblib.dump(dbscan_model, MODELS_DIR / "customer_dbscan_model.joblib")
    joblib.dump(pca, MODELS_DIR / "customer_pca_model.joblib")

    customer_rfm.to_csv(PROCESSED_DIR / "customer_cluster_labels.csv", index=False)
    persona_frame.to_csv(PROCESSED_DIR / "customer_personas.csv", index=False)

    cluster_profile = customer_rfm.groupby("kmeans_cluster")[feature_cols].mean().round(2)
    cluster_profile.to_csv(PROCESSED_DIR / "cluster_profile_kmeans.csv")

    dbscan_profile = customer_rfm.groupby("dbscan_cluster")[feature_cols].mean().round(2)
    dbscan_profile.to_csv(PROCESSED_DIR / "cluster_profile_dbscan.csv")

    kmeans_elbow_table = pd.DataFrame(kmeans_grid)
    kmeans_elbow_table.to_csv(PROCESSED_DIR / "kmeans_elbow_metrics.csv", index=False)
    dbscan_grid_table = pd.DataFrame(dbscan_grid)
    dbscan_grid_table.to_csv(PROCESSED_DIR / "dbscan_grid_metrics.csv", index=False)

    customer_rfm.attrs["kmeans_elbow_inertia"] = elbow_inertia
    save_visualizations(customer_rfm)

    best_kmeans_score = silhouette_score(scaled_features, customer_rfm["kmeans_cluster"])
    valid_dbscan_mask = customer_rfm["dbscan_cluster"] != -1
    if valid_dbscan_mask.sum() > 10 and len(set(customer_rfm.loc[valid_dbscan_mask, "dbscan_cluster"])) > 1:
        best_dbscan_score = silhouette_score(scaled_features[valid_dbscan_mask], customer_rfm.loc[valid_dbscan_mask, "dbscan_cluster"])
    else:
        best_dbscan_score = float("nan")

    save_cluster_report(customer_rfm, best_kmeans_score, best_dbscan_score, dbscan_model, PROCESSED_DIR / "customer_segmentation_report.md")
    return customer_rfm


if __name__ == "__main__":
    run_customer_segmentation_pipeline()
