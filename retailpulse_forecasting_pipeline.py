from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

from retailpulse_feature_pipeline import FIGURES_DIR, PROCESSED_DIR, ensure_output_dirs


def load_processed_dataset(file_name: str) -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / file_name, parse_dates=["date"])


def adf_summary(series: pd.Series) -> dict[str, object]:
    clean_series = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean_series) < 20:
        return {
            "series_length": len(clean_series),
            "adf_statistic": np.nan,
            "p_value": np.nan,
            "lags_used": np.nan,
            "observations": np.nan,
            "stationary": False,
        }

    result = adfuller(clean_series, autolag="AIC")
    return {
        "series_length": len(clean_series),
        "adf_statistic": result[0],
        "p_value": result[1],
        "lags_used": result[2],
        "observations": result[3],
        "critical_1%": result[4]["1%"],
        "critical_5%": result[4]["5%"],
        "critical_10%": result[4]["10%"],
        "stationary": result[1] < 0.05,
    }


def safe_seasonal_decompose(series: pd.Series, period: int) -> seasonal_decompose | None:
    clean_series = pd.to_numeric(series, errors="coerce").fillna(0)
    if len(clean_series) < period * 2:
        return None
    return seasonal_decompose(clean_series, model="additive", period=period, extrapolate_trend="freq")


def save_daily_visuals(daily_sales: pd.DataFrame) -> None:
    plt.figure(figsize=(14, 6))
    plt.plot(daily_sales["date"], daily_sales["total_price"], color="#1f77b4", linewidth=1.5)
    plt.title("Daily Sales Trend")
    plt.xlabel("Date")
    plt.ylabel("Revenue")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "daily_sales_trend.png", dpi=160)
    plt.close()

    plt.figure(figsize=(14, 6))
    plt.plot(daily_sales["date"], daily_sales["rolling_mean_7"], color="#ff7f0e", linewidth=2)
    plt.title("7-Day Rolling Mean of Daily Sales")
    plt.xlabel("Date")
    plt.ylabel("Revenue")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "daily_sales_rolling_mean.png", dpi=160)
    plt.close()


def save_decomposition_plot(series: pd.Series, period: int, title: str, file_name: str) -> dict[str, object]:
    decomposition = safe_seasonal_decompose(series, period=period)
    if decomposition is None:
        return {"title": title, "status": "insufficient_data"}

    fig = decomposition.plot()
    fig.set_size_inches(14, 9)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / file_name, dpi=160)
    plt.close(fig)

    return {
        "title": title,
        "status": "saved",
        "trend_mean": float(pd.Series(decomposition.trend).dropna().mean()),
        "seasonal_mean": float(pd.Series(decomposition.seasonal).dropna().mean()),
        "resid_std": float(pd.Series(decomposition.resid).dropna().std()),
    }


def build_forecasting_report(daily_stats: dict[str, object], weekly_stats: dict[str, object], product_stats: dict[str, object], report_path: Path) -> None:
    def _md(frame: pd.DataFrame) -> str:
        try:
            return frame.to_markdown(index=False)
        except ImportError:
            return frame.to_csv(index=False)

    lines = [
        "# RetailPulse Forecasting Readiness Report",
        "",
        "## Stationarity Results",
        "",
        _md(pd.DataFrame([daily_stats, weekly_stats, product_stats])),
        "",
        "## Interpretation",
        "",
        "- Daily and weekly revenue series should be inspected for trend and seasonality before model selection.",
        "- A p-value below 0.05 indicates stationarity and supports direct ARIMA-style modeling with fewer differencing steps.",
        "- Non-stationary series should be differenced or transformed before forecasting.",
        "- Product-level demand is highly sparse, so the highest-revenue products are the best candidates for an initial forecast benchmark.",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_forecasting_pipeline() -> pd.DataFrame:
    ensure_output_dirs()
    daily_sales = load_processed_dataset("daily_sales_forecasting.csv")
    weekly_sales = load_processed_dataset("weekly_sales_forecasting.csv")
    product_demand = pd.read_csv(PROCESSED_DIR / "product_daily_demand.csv", parse_dates=["date"])

    save_daily_visuals(daily_sales)

    daily_stats = adf_summary(daily_sales["total_price"])
    weekly_stats = adf_summary(weekly_sales["total_price"])
    top_product = product_demand.sort_values(["total_price", "quantity"], ascending=False).groupby("StockCode").head(1)
    top_stock_code = top_product.iloc[0]["StockCode"] if not top_product.empty else None
    top_product_series = product_demand.loc[product_demand["StockCode"] == top_stock_code, "quantity"] if top_stock_code is not None else pd.Series(dtype=float)
    product_stats = adf_summary(top_product_series)

    daily_decomposition = save_decomposition_plot(
        daily_sales.set_index("date")["total_price"],
        period=7,
        title="Daily Sales Seasonal Decomposition",
        file_name="daily_sales_decomposition.png",
    )

    weekly_decomposition = save_decomposition_plot(
        weekly_sales.set_index("date")["total_price"],
        period=4,
        title="Weekly Sales Seasonal Decomposition",
        file_name="weekly_sales_decomposition.png",
    )

    if top_stock_code is not None:
        top_product_frame = product_demand.loc[product_demand["StockCode"] == top_stock_code].set_index("date")
        save_decomposition_plot(
            top_product_frame["quantity"],
            period=7,
            title=f"Top Product Demand Seasonal Decomposition ({top_stock_code})",
            file_name="top_product_decomposition.png",
        )

    stationarity_table = pd.DataFrame([
        {"series": "daily_sales_total_price", **daily_stats},
        {"series": "weekly_sales_total_price", **weekly_stats},
        {"series": f"top_product_{top_stock_code}_quantity", **product_stats},
    ])
    stationarity_table.to_csv(PROCESSED_DIR / "forecasting_stationarity_results.csv", index=False)

    decomposition_summary = pd.DataFrame([
        {"series": "daily_sales_total_price", **daily_decomposition},
        {"series": "weekly_sales_total_price", **weekly_decomposition},
    ])
    decomposition_summary.to_csv(PROCESSED_DIR / "forecasting_decomposition_summary.csv", index=False)

    build_forecasting_report(daily_stats, weekly_stats, product_stats, PROCESSED_DIR / "forecasting_interpretation_report.md")
    return stationarity_table


if __name__ == "__main__":
    run_forecasting_pipeline()
