from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from retailpulse_feature_pipeline import FIGURES_DIR, MODELS_DIR, PROCESSED_DIR, ensure_output_dirs


@dataclass
class LSTMRunResult:
    metrics: dict[str, float]
    predictions_path: Path
    report_path: Path
    checkpoint_path: Path
    state_dict_path: Path
    scaler_path: Path
    config_path: Path


class SequenceDataset(Dataset):
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        return self.sequences[index], self.targets[index]


class RetailDemandLSTM(pl.LightningModule):
    def __init__(self, input_size: int = 1, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2, learning_rate: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        output, _ = self.lstm(x)
        last_hidden = output[:, -1, :]
        prediction = self.regressor(last_hidden)
        return prediction.squeeze(-1)

    def _shared_step(self, batch, stage: str):
        sequences, targets = batch
        predictions = self(sequences)
        loss = self.loss_fn(predictions, targets)
        self.log(f"{stage}_loss", loss, prog_bar=(stage != "train"), on_epoch=True, on_step=False)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


def load_daily_sales() -> pd.DataFrame:
    frame = pd.read_csv(PROCESSED_DIR / "daily_sales_forecasting.csv", parse_dates=["date"])
    frame = frame[["date", "total_price"]].dropna().sort_values("date").reset_index(drop=True)
    return frame


def build_sequence_arrays(values: np.ndarray, dates: pd.Series, sequence_length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sequences: list[np.ndarray] = []
    targets: list[float] = []
    target_dates: list[pd.Timestamp] = []
    for target_index in range(sequence_length, len(values)):
        sequences.append(values[target_index - sequence_length : target_index].reshape(sequence_length, 1))
        targets.append(values[target_index])
        target_dates.append(pd.to_datetime(dates.iloc[target_index]))
    return np.array(sequences), np.array(targets), np.array(target_dates)


def split_sequences(sequences: np.ndarray, targets: np.ndarray, target_dates: np.ndarray, sequence_length: int, train_ratio: float = 0.7, val_ratio: float = 0.15):
    total_targets = len(targets)
    train_cutoff = int(total_targets * train_ratio)
    val_cutoff = int(total_targets * (train_ratio + val_ratio))

    train_sequences = sequences[:train_cutoff]
    train_targets = targets[:train_cutoff]
    train_dates = target_dates[:train_cutoff]

    val_sequences = sequences[train_cutoff:val_cutoff]
    val_targets = targets[train_cutoff:val_cutoff]
    val_dates = target_dates[train_cutoff:val_cutoff]

    test_sequences = sequences[val_cutoff:]
    test_targets = targets[val_cutoff:]
    test_dates = target_dates[val_cutoff:]

    return (train_sequences, train_targets, train_dates), (val_sequences, val_targets, val_dates), (test_sequences, test_targets, test_dates)


def inverse_transform(values: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    reshaped = values.reshape(-1, 1)
    return scaler.inverse_transform(reshaped).reshape(-1)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    denominator = np.clip(np.abs(actual), 1e-8, None)
    mape = float(np.mean(np.abs((actual - predicted) / denominator)) * 100)
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    mae = float(mean_absolute_error(actual, predicted))
    return {"MAPE": mape, "RMSE": rmse, "MAE": mae}


def prepare_dataloaders(sequence_length: int = 30, batch_size: int = 32):
    sales = load_daily_sales()
    series = sales["total_price"].astype(float).to_numpy()

    train_cutoff = int(len(series) * 0.7)
    scaler = MinMaxScaler()
    scaler.fit(series[:train_cutoff].reshape(-1, 1))
    scaled_series = scaler.transform(series.reshape(-1, 1)).reshape(-1)

    sequences, targets, target_dates = build_sequence_arrays(scaled_series, sales["date"], sequence_length)
    (train_sequences, train_targets, train_dates), (val_sequences, val_targets, val_dates), (test_sequences, test_targets, test_dates) = split_sequences(
        sequences, targets, target_dates, sequence_length
    )

    train_loader = DataLoader(SequenceDataset(train_sequences, train_targets), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SequenceDataset(val_sequences, val_targets), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(SequenceDataset(test_sequences, test_targets), batch_size=batch_size, shuffle=False)

    return {
        "sales": sales,
        "series": series,
        "scaler": scaler,
        "scaled_series": scaled_series,
        "train": (train_sequences, train_targets, train_dates),
        "val": (val_sequences, val_targets, val_dates),
        "test": (test_sequences, test_targets, test_dates),
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "sequence_length": sequence_length,
    }


def train_lstm_model(dataloaders: dict[str, Any]) -> tuple[RetailDemandLSTM, pl.Trainer]:
    pl.seed_everything(42, workers=True)
    model = RetailDemandLSTM(input_size=1, hidden_size=64, num_layers=2, dropout=0.2, learning_rate=1e-3)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=str(MODELS_DIR),
        filename="lstm_best_model",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    early_stopping = pl.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")
    trainer = pl.Trainer(
        max_epochs=40,
        callbacks=[checkpoint_callback, early_stopping],
        logger=False,
        enable_checkpointing=True,
        enable_progress_bar=False,
        accelerator="cpu",
        deterministic=True,
    )
    trainer.fit(model, dataloaders["train_loader"], dataloaders["val_loader"])
    best_model = RetailDemandLSTM.load_from_checkpoint(checkpoint_callback.best_model_path)
    return best_model, trainer


def evaluate_model(model: RetailDemandLSTM, dataloaders: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, float]]:
    scaler: MinMaxScaler = dataloaders["scaler"]
    test_loader = dataloaders["test_loader"]
    test_sequences, test_targets, test_dates = dataloaders["test"]

    model.eval()
    predictions_scaled: list[float] = []
    with torch.no_grad():
        for batch_sequences, _ in test_loader:
            batch_predictions = model(batch_sequences).cpu().numpy()
            predictions_scaled.extend(batch_predictions.tolist())

    actual_scaled = np.array(test_targets)
    actual = inverse_transform(actual_scaled, scaler)
    predicted = inverse_transform(np.array(predictions_scaled), scaler)
    metrics = compute_metrics(actual, predicted)

    prediction_frame = pd.DataFrame({
        "date": pd.to_datetime(test_dates),
        "actual": actual,
        "predicted": predicted,
        "error": actual - predicted,
    })
    prediction_frame["absolute_error"] = prediction_frame["error"].abs()
    return prediction_frame, metrics


def plot_predictions(prediction_frame: pd.DataFrame) -> Path:
    plt.figure(figsize=(14, 6))
    plt.plot(prediction_frame["date"], prediction_frame["actual"], label="Actual", color="#1f77b4", linewidth=2)
    plt.plot(prediction_frame["date"], prediction_frame["predicted"], label="Predicted", color="#ff7f0e", linewidth=2)
    plt.title("LSTM Test Set Forecast vs Actuals")
    plt.xlabel("Date")
    plt.ylabel("Revenue")
    plt.legend()
    plt.tight_layout()
    output_path = FIGURES_DIR / "lstm_test_predictions.png"
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def build_report(metrics: dict[str, float], prediction_frame: pd.DataFrame, report_path: Path) -> None:
    summary = prediction_frame.tail(10).copy()
    summary["date"] = pd.to_datetime(summary["date"]).dt.strftime("%Y-%m-%d")
    lines = [
        "# RetailPulse LSTM Forecasting Evaluation",
        "",
        "## Test Metrics",
        "",
        pd.DataFrame([metrics]).to_markdown(index=False),
        "",
        "## Recent Predictions",
        "",
        summary[["date", "actual", "predicted", "absolute_error"]].to_markdown(index=False),
        "",
        "## Business Interpretation",
        "",
        "- The test split is strictly chronological, so the reported error reflects genuine forward-looking performance.",
        "- The LSTM captures nonlinear sequence effects that a linear baseline may miss, but it still depends heavily on stable demand patterns.",
        "- If the model underperforms Prophet, the business should prefer the simpler model for deployment until more exogenous signals are available.",
        "- Re-training should happen on a rolling basis because retail demand shifts with promotions and seasonality.",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_lstm_pipeline() -> LSTMRunResult:
    ensure_output_dirs()
    dataloaders = prepare_dataloaders(sequence_length=30, batch_size=32)
    model, trainer = train_lstm_model(dataloaders)
    prediction_frame, metrics = evaluate_model(model, dataloaders)

    predictions_path = PROCESSED_DIR / "lstm_test_predictions.csv"
    report_path = PROCESSED_DIR / "lstm_evaluation_report.md"
    checkpoint_path = MODELS_DIR / "lstm_best_model.ckpt"
    state_dict_path = MODELS_DIR / "lstm_model_state_dict.pt"
    scaler_path = MODELS_DIR / "lstm_scaler.pkl"
    config_path = MODELS_DIR / "lstm_config.json"

    prediction_frame.to_csv(predictions_path, index=False)
    build_report(metrics, prediction_frame, report_path)
    plot_predictions(prediction_frame)

    torch.save(model.state_dict(), checkpoint_path)
    torch.save(model.state_dict(), state_dict_path)
    with open(scaler_path, "wb") as handle:
        pickle.dump(dataloaders["scaler"], handle)

    config_payload = {
        "sequence_length": dataloaders["sequence_length"],
        "input_size": 1,
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.2,
        "learning_rate": 0.001,
    }
    config_path.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    pd.DataFrame([metrics]).to_csv(PROCESSED_DIR / "lstm_metrics.csv", index=False)

    return LSTMRunResult(
        metrics=metrics,
        predictions_path=predictions_path,
        report_path=report_path,
        checkpoint_path=checkpoint_path,
        state_dict_path=state_dict_path,
        scaler_path=scaler_path,
        config_path=config_path,
    )


if __name__ == "__main__":
    run_lstm_pipeline()
