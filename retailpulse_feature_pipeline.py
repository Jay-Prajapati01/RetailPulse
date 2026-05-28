from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT_DIR = Path(__file__).resolve().parent
DATA_FILE = ROOT_DIR / "online_retail_II.xlsx"
PROCESSED_DIR = ROOT_DIR / "processed"
MODELS_DIR = PROCESSED_DIR / "models"
FIGURES_DIR = PROCESSED_DIR / "figures"

RAW_COLUMNS = [
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country",
]


@dataclass
class PipelineOutputs:
    raw_transactions: pd.DataFrame
    cleaned_transactions: pd.DataFrame
    customer_base: pd.DataFrame
    rfm_table: pd.DataFrame
    encoded_customer_matrix: pd.DataFrame
    daily_sales: pd.DataFrame
    weekly_sales: pd.DataFrame
    product_demand: pd.DataFrame


def ensure_output_dirs() -> None:
    PROCESSED_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [column.strip().replace(" ", "") for column in frame.columns]
    frame = frame.rename(
        columns={
            "Invoice": "InvoiceNo",
            "Price": "UnitPrice",
            "CustomerID": "CustomerID",
        }
    )
    return frame


def load_tabular_dataset(file_path: Path) -> pd.DataFrame:
    if file_path.suffix.lower() in {".xlsx", ".xls"}:
        workbook = pd.ExcelFile(file_path, engine="openpyxl")
        frames: list[pd.DataFrame] = []
        for sheet_name in workbook.sheet_names:
            frame = pd.read_excel(
                workbook,
                sheet_name=sheet_name,
                usecols=RAW_COLUMNS,
                dtype={
                    "Invoice": "string",
                    "StockCode": "string",
                    "Description": "string",
                    "Customer ID": "string",
                    "Country": "string",
                },
                parse_dates=["InvoiceDate"],
                engine="openpyxl",
            )
            frame["SourceSheet"] = sheet_name
            frames.append(frame)
        combined = pd.concat(frames, ignore_index=True)
        return normalize_columns(combined)

    encodings = ["utf-8-sig", "utf-8", "latin1", "cp1252"]
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            frame = pd.read_csv(file_path, usecols=RAW_COLUMNS, encoding=encoding)
            return normalize_columns(frame)
        except Exception as exc:  # pragma: no cover - fallback path
            last_error = exc
    if last_error is None:
        raise ValueError(f"Could not load dataset: {file_path}")
    raise last_error


def coerce_invoice_date(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.notna().any():
        return parsed

    numeric_dates = pd.to_numeric(series, errors="coerce")
    return pd.to_datetime(numeric_dates, unit="d", origin="1899-12-30", errors="coerce")


def clean_transactions(raw_df: pd.DataFrame) -> pd.DataFrame:
    frame = raw_df.copy()
    frame["InvoiceNo"] = frame["InvoiceNo"].astype("string").str.strip()
    frame["StockCode"] = frame["StockCode"].astype("string").str.strip()
    frame["Description"] = frame["Description"].astype("string").str.strip()
    frame["Country"] = frame["Country"].astype("string").str.strip()
    frame["InvoiceDate"] = coerce_invoice_date(frame["InvoiceDate"])
    frame["Quantity"] = pd.to_numeric(frame["Quantity"], errors="coerce")
    frame["UnitPrice"] = pd.to_numeric(frame["UnitPrice"], errors="coerce")
    frame["CustomerID"] = pd.to_numeric(frame["CustomerID"], errors="coerce").astype("Int64")

    frame["total_price"] = frame["Quantity"] * frame["UnitPrice"]
    frame["is_cancellation"] = frame["InvoiceNo"].str.startswith("C", na=False)
    frame["has_customer_id"] = frame["CustomerID"].notna()

    cleaned = frame.loc[
        frame["InvoiceDate"].notna()
        & frame["Quantity"].gt(0)
        & frame["UnitPrice"].gt(0)
        & ~frame["is_cancellation"]
    ].copy()

    cleaned["InvoiceMonth"] = cleaned["InvoiceDate"].dt.to_period("M").dt.to_timestamp()
    cleaned["InvoiceWeek"] = cleaned["InvoiceDate"].dt.to_period("W").dt.start_time
    cleaned["InvoiceDateOnly"] = cleaned["InvoiceDate"].dt.normalize()
    return cleaned


def build_customer_base(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    customer_df = cleaned_df.loc[cleaned_df["CustomerID"].notna()].copy()
    customer_df["CustomerID"] = customer_df["CustomerID"].astype("Int64")

    grouped = customer_df.groupby("CustomerID")
    customer_base = grouped.agg(
        first_purchase=("InvoiceDate", "min"),
        last_purchase=("InvoiceDate", "max"),
        frequency=("InvoiceNo", "nunique"),
        line_items=("InvoiceNo", "size"),
        total_units=("Quantity", "sum"),
        total_revenue=("total_price", "sum"),
        avg_unit_price=("UnitPrice", "mean"),
        avg_basket_size=("Quantity", "mean"),
        country=("Country", lambda values: values.dropna().mode().iat[0] if not values.dropna().empty else pd.NA),
    ).reset_index()

    analysis_date = customer_df["InvoiceDate"].max() + pd.Timedelta(days=1)
    customer_base["recency_days"] = (analysis_date - customer_base["last_purchase"]).dt.days
    customer_base["tenure_days"] = (customer_base["last_purchase"] - customer_base["first_purchase"]).dt.days
    customer_base["orders_per_day"] = customer_base["frequency"] / customer_base["tenure_days"].replace(0, np.nan)
    customer_base["revenue_per_order"] = customer_base["total_revenue"] / customer_base["frequency"].replace(0, np.nan)
    customer_base["units_per_order"] = customer_base["total_units"] / customer_base["frequency"].replace(0, np.nan)
    customer_base["days_since_first_purchase"] = (analysis_date - customer_base["first_purchase"]).dt.days
    customer_base["days_since_last_purchase"] = customer_base["recency_days"]
    customer_base["purchase_span_days"] = (
        customer_base["last_purchase"] - customer_base["first_purchase"]
    ).dt.days
    customer_base["is_inactive_90d"] = customer_base["recency_days"] >= 90
    customer_base["is_high_value"] = customer_base["total_revenue"] >= customer_base["total_revenue"].quantile(0.75)
    customer_base["is_repeat_customer"] = customer_base["frequency"] > 1
    customer_base["source_row_count"] = grouped.size().values
    return customer_base


def build_rfm_table(customer_base: pd.DataFrame) -> pd.DataFrame:
    rfm = customer_base[["CustomerID", "recency_days", "frequency", "total_revenue", "country"]].copy()
    rfm = rfm.rename(
        columns={
            "recency_days": "Recency",
            "frequency": "Frequency",
            "total_revenue": "Monetary",
            "country": "Country",
        }
    )

    rfm["R_Score"] = pd.qcut(rfm["Recency"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1])
    rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    rfm["M_Score"] = pd.qcut(rfm["Monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    rfm["RFM_Score"] = rfm[["R_Score", "F_Score", "M_Score"]].astype(str).agg("".join, axis=1)
    rfm["RFM_Total"] = rfm[["R_Score", "F_Score", "M_Score"]].astype(int).sum(axis=1)
    return rfm


def build_customer_model_matrix(customer_base: pd.DataFrame) -> pd.DataFrame:
    modeling_frame = customer_base[[
        "CustomerID",
        "recency_days",
        "frequency",
        "total_revenue",
        "avg_unit_price",
        "avg_basket_size",
        "orders_per_day",
        "revenue_per_order",
        "units_per_order",
        "country",
    ]].copy()

    numeric_features = [
        "recency_days",
        "frequency",
        "total_revenue",
        "avg_unit_price",
        "avg_basket_size",
        "orders_per_day",
        "revenue_per_order",
        "units_per_order",
    ]
    categorical_features = ["country"]

    transformer = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    matrix = transformer.fit_transform(modeling_frame)
    feature_names = transformer.get_feature_names_out()
    encoded = pd.DataFrame(matrix, columns=feature_names, index=modeling_frame.index)
    encoded.insert(0, "CustomerID", modeling_frame["CustomerID"].values)
    return encoded


def add_lag_and_rolling_features(frame: pd.DataFrame, value_column: str, lags: Sequence[int], windows: Sequence[int]) -> pd.DataFrame:
    frame = frame.sort_values("date").copy()
    for lag in lags:
        frame[f"lag_{lag}"] = frame[value_column].shift(lag)
    for window in windows:
        frame[f"rolling_mean_{window}"] = frame[value_column].rolling(window=window, min_periods=1).mean().shift(1)
        frame[f"rolling_std_{window}"] = frame[value_column].rolling(window=window, min_periods=2).std().shift(1)
    return frame


def build_daily_sales(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    daily = (
        cleaned_df.groupby("InvoiceDateOnly", as_index=False)
        .agg(
            total_price=("total_price", "sum"),
            quantity=("Quantity", "sum"),
            invoice_count=("InvoiceNo", "nunique"),
            customer_count=("CustomerID", "nunique"),
        )
        .rename(columns={"InvoiceDateOnly": "date"})
    )

    full_date_index = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    daily = daily.set_index("date").reindex(full_date_index, fill_value=0).rename_axis("date").reset_index()
    daily["day_of_week"] = daily["date"].dt.dayofweek
    daily["is_weekend"] = daily["day_of_week"].isin([5, 6]).astype(int)
    daily["month"] = daily["date"].dt.month
    daily["year"] = daily["date"].dt.year
    daily["week"] = daily["date"].dt.isocalendar().week.astype(int)
    daily = add_lag_and_rolling_features(daily, "total_price", lags=[1, 7, 14, 28], windows=[7, 30])
    return daily


def build_weekly_sales(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    weekly = (
        cleaned_df.groupby("InvoiceWeek", as_index=False)
        .agg(
            total_price=("total_price", "sum"),
            quantity=("Quantity", "sum"),
            invoice_count=("InvoiceNo", "nunique"),
            customer_count=("CustomerID", "nunique"),
        )
        .rename(columns={"InvoiceWeek": "date"})
    )

    full_date_index = pd.date_range(weekly["date"].min(), weekly["date"].max(), freq="W-MON")
    weekly = weekly.set_index("date").reindex(full_date_index, fill_value=0).rename_axis("date").reset_index()
    weekly["week_of_year"] = weekly["date"].dt.isocalendar().week.astype(int)
    weekly["month"] = weekly["date"].dt.month
    weekly["year"] = weekly["date"].dt.year
    weekly = add_lag_and_rolling_features(weekly, "total_price", lags=[1, 2, 4, 8], windows=[4, 12])
    return weekly


def build_product_demand(cleaned_df: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
    top_products = (
        cleaned_df.groupby(["StockCode", "Description"], as_index=False)
        .agg(total_price=("total_price", "sum"), total_units=("Quantity", "sum"))
        .sort_values("total_price", ascending=False)
        .head(top_n)
    )

    demand_frames: list[pd.DataFrame] = []
    for _, product_row in top_products.iterrows():
        product_mask = cleaned_df["StockCode"].eq(product_row["StockCode"])
        product_series = (
            cleaned_df.loc[product_mask]
            .groupby("InvoiceDateOnly", as_index=False)
            .agg(
                total_price=("total_price", "sum"),
                quantity=("Quantity", "sum"),
                invoice_count=("InvoiceNo", "nunique"),
            )
            .rename(columns={"InvoiceDateOnly": "date"})
        )
        full_date_index = pd.date_range(cleaned_df["InvoiceDateOnly"].min(), cleaned_df["InvoiceDateOnly"].max(), freq="D")
        product_series = product_series.set_index("date").reindex(full_date_index, fill_value=0).rename_axis("date").reset_index()
        product_series["StockCode"] = product_row["StockCode"]
        product_series["Description"] = product_row["Description"]
        product_series["rank_by_revenue"] = int(len(demand_frames) + 1)
        product_series = add_lag_and_rolling_features(product_series, "quantity", lags=[1, 7, 14], windows=[7, 30])
        product_series = add_lag_and_rolling_features(product_series, "total_price", lags=[1, 7, 14], windows=[7, 30])
        demand_frames.append(product_series)

    return pd.concat(demand_frames, ignore_index=True)


def build_churn_features(customer_base: pd.DataFrame) -> pd.DataFrame:
    churn = customer_base[["CustomerID", "recency_days", "frequency", "total_revenue", "orders_per_day", "is_inactive_90d", "is_high_value"]].copy()
    churn["churn_risk_score"] = (
        churn["recency_days"].rank(pct=True) * 0.5
        + (1 - churn["frequency"].rank(pct=True)) * 0.3
        + (1 - churn["total_revenue"].rank(pct=True)) * 0.2
    )
    churn["churn_label"] = np.select(
        [churn["recency_days"] >= 180, churn["recency_days"] >= 90],
        ["Lost", "At Risk"],
        default="Active",
    )
    return churn


def save_dataframe(frame: pd.DataFrame, file_name: str) -> Path:
    output_path = PROCESSED_DIR / file_name
    frame.to_csv(output_path, index=False)
    return output_path


def build_pipeline_outputs(data_file: Path = DATA_FILE) -> PipelineOutputs:
    ensure_output_dirs()
    raw_transactions = load_tabular_dataset(data_file)
    cleaned_transactions = clean_transactions(raw_transactions)
    customer_base = build_customer_base(cleaned_transactions)
    rfm_table = build_rfm_table(customer_base)
    encoded_customer_matrix = build_customer_model_matrix(customer_base)
    daily_sales = build_daily_sales(cleaned_transactions)
    weekly_sales = build_weekly_sales(cleaned_transactions)
    product_demand = build_product_demand(cleaned_transactions)
    return PipelineOutputs(
        raw_transactions=raw_transactions,
        cleaned_transactions=cleaned_transactions,
        customer_base=customer_base,
        rfm_table=rfm_table,
        encoded_customer_matrix=encoded_customer_matrix,
        daily_sales=daily_sales,
        weekly_sales=weekly_sales,
        product_demand=product_demand,
    )


def run_and_save_pipeline(data_file: Path = DATA_FILE) -> PipelineOutputs:
    outputs = build_pipeline_outputs(data_file)

    save_dataframe(outputs.raw_transactions, "retail_transactions_raw.csv")
    save_dataframe(outputs.cleaned_transactions, "retail_transactions_cleaned.csv")
    save_dataframe(outputs.customer_base, "customer_base_features.csv")
    save_dataframe(outputs.rfm_table, "customer_rfm.csv")
    save_dataframe(outputs.encoded_customer_matrix, "customer_encoded_features.csv")
    save_dataframe(outputs.daily_sales, "daily_sales_forecasting.csv")
    save_dataframe(outputs.weekly_sales, "weekly_sales_forecasting.csv")
    save_dataframe(outputs.product_demand, "product_daily_demand.csv")

    churn_features = build_churn_features(outputs.customer_base)
    save_dataframe(churn_features, "customer_churn_features.csv")

    return outputs


if __name__ == "__main__":
    run_and_save_pipeline()
