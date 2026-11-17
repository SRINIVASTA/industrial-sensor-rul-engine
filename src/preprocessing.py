import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sktime.transformations.series.summarize import WindowSummarizer

SENSOR_COLS = ["temp_c", "vibration_mms", "pressure_bar", "rpm"]


def extract_sktime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Uses sktime WindowSummarizer to compute rolling sensor statistics."""
    processed_dfs = []

    kwargs = {
        "lag_feature": {
            "lag":,
            "mean": [[1, 5]],
            "std": [[1, 5]],
        }
    }
    transformer = WindowSummarizer(**kwargs, target_cols=SENSOR_COLS)

    for _, group in df.groupby("machine_id"):
        group = group.sort_values("cycle").reset_index(drop=True)
        features = transformer.fit_transform(group[SENSOR_COLS])
        combined = pd.concat([group[["machine_id", "cycle", "RUL"]], features], axis=1)
        processed_dfs.append(combined.bfill().ffill())

    return pd.concat(processed_dfs, ignore_index=True)


def create_sequences(df: pd.DataFrame, seq_length: int = 20):
    """Creates 3D sliding windows (samples, time_steps, features) for PyTorch LSTM."""
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[SENSOR_COLS] = scaler.fit_transform(df[SENSOR_COLS])

    X, y = [], []
    for _, group in df_scaled.groupby("machine_id"):
        data = group[SENSOR_COLS].values
        targets = group["RUL"].values
        for i in range(len(data) - seq_length):
            X.append(data[i : i + seq_length])
            y.append(targets[i + seq_length])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), scaler
