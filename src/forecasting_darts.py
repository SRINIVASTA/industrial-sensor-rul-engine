import gc
from darts import TimeSeries
from darts.metrics import mape
from darts.models import ExponentialSmoothing, NBEATSModel
import pandas as pd
import torch


def forecast_sensor_drift(
    df: pd.DataFrame, machine_id: int = 1, horizon: int = 15
):
    """Compares Statistical vs Deep Learning forecasting on vibration telemetry."""
    machine_df = df[df["machine_id"] == machine_id].copy()
    series = TimeSeries.from_dataframe(
        machine_df, time_col="cycle", value_cols="vibration_mms"
    )

    train, val = series[:-horizon], series[-horizon:]

    # 1. Statistical Baseline (Exponential Smoothing)
    stat_model = ExponentialSmoothing()
    stat_model.fit(train)
    stat_pred = stat_model.predict(horizon)

    # 2. Lightweight Deep Learning Model (N-BEATS) - ~110K params instead of 6.2M
    dl_model = NBEATSModel(
        input_chunk_length=24,
        output_chunk_length=horizon,
        num_stacks=4,
        num_blocks=1,
        num_layers=2,
        layer_widths=64,
        n_epochs=25,
        batch_size=32,
        random_state=42,
        pl_trainer_kwargs={
            "accelerator": "cpu",
            "enable_progress_bar": False,
            "enable_checkpointing": False,
            "logger": False,
        },
    )
    dl_model.fit(train)
    dl_pred = dl_model.predict(horizon)

    result = {
        "actual": val.values().flatten(),
        "stat_forecast": stat_pred.values().flatten(),
        "nbeats_forecast": dl_pred.values().flatten(),
        "stat_mape": round(mape(val, stat_pred), 2),
        "nbeats_mape": round(mape(val, dl_pred), 2),
    }

    # Free model weights and PyTorch buffers immediately
    del dl_model, stat_model
    gc.collect()

    return result
