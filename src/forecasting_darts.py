import pandas as pd
from darts import TimeSeries
from darts.models import ExponentialSmoothing, NBEATSModel
from darts.metrics import mape

def forecast_sensor_drift(df: pd.DataFrame, machine_id: int = 1, horizon: int = 15):
    """Compares Statistical vs Deep Learning forecasting on vibration telemetry."""
    machine_df = df[df["machine_id"] == machine_id].copy()
    series = TimeSeries.from_dataframe(machine_df, time_col="cycle", value_cols="vibration_mms")

    train, val = series[:-horizon], series[-horizon:]

    # 1. Statistical Baseline (Exponential Smoothing)
    stat_model = ExponentialSmoothing()
    stat_model.fit(train)
    stat_pred = stat_model.predict(horizon)

    # 2. Deep Learning Model (N-BEATS)
    dl_model = NBEATSModel(
        input_chunk_length=24,
        output_chunk_length=horizon,
        n_epochs=30,
        random_state=42,
        pl_trainer_kwargs={"accelerator": "cpu", "enable_progress_bar": False}
    )
    dl_model.fit(train)
    dl_pred = dl_model.predict(horizon)

    return {
        "actual": val.values().flatten(),
        "stat_forecast": stat_pred.values().flatten(),
        "nbeats_forecast": dl_pred.values().flatten(),
        "stat_mape": round(mape(val, stat_pred), 2),
        "nbeats_mape": round(mape(val, dl_pred), 2),
    }
