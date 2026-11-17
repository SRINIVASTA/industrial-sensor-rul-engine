import streamlit as st
import torch
import plotly.graph_objects as go
from src.data_simulator import generate_sensor_data
from src.preprocessing import extract_sktime_features, create_sequences
from src.forecasting_darts import forecast_sensor_drift
from src.rul_pytorch import train_rul_model

st.set_page_config(
    page_title="Industrial Sensor RUL & Drift Engine",
    page_icon="⚙️",
    layout="wide",
)

st.title("⚙️ Industrial Predictive Maintenance & Sensor Drift Engine")
st.caption(
    "End-to-End Applied ML Pipeline • PyTorch (LSTM) • Darts (N-BEATS & Exponential Smoothing) • sktime"
)


@st.cache_data
def load_telemetry_data():
    return generate_sensor_data(n_machines=15)


@st.cache_data
def run_cached_darts(df, machine_id: int, horizon: int = 15):
    return forecast_sensor_drift(df, machine_id=machine_id, horizon=horizon)


@st.cache_resource
def get_trained_lstm(df, test_machine_id: int):
    _ = extract_sktime_features(df)
    X_train, y_train, _ = create_sequences(
        df[df["machine_id"] != test_machine_id], seq_length=20
    )
    return train_rul_model(X_train, y_train, epochs=20)


# Load simulated industrial sensor dataset
df = load_telemetry_data()

# Sidebar controls
st.sidebar.header("🎛️ Asset Controls")
machine_id = st.sidebar.selectbox(
    "Select Turbine Asset ID", sorted(df["machine_id"].unique())
)
horizon = st.sidebar.slider("Darts Forecast Horizon (Cycles)", 10, 25, 15)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Tech Stack:**\n"
    "- `sktime`: Rolling window feature extraction\n"
    "- `Darts`: Exponential Smoothing vs. N-BEATS\n"
    "- `PyTorch`: 2-Layer LSTM with Huber Loss\n"
    "- `Plotly` & `Streamlit`: Real-time diagnostics"
)

# Filter selected turbine data
asset_df = df[df["machine_id"] == machine_id].sort_values("cycle")

# Top KPI Row
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Operating Cycles", f"{int(asset_df['cycle'].max())} cycles")
k2.metric("Peak Vibration", f"{asset_df['vibration_mms'].max():.2f} mm/s")
k3.metric("Peak Core Temp", f"{asset_df['temp_c'].max():.1f} °C")
k4.metric("Final Pressure", f"{asset_df['pressure_bar'].iloc[-1]:.2f} bar")

# Expandable Raw Multivariate Sensor Telemetry
with st.expander("📊 View Raw Multivariate Sensor Telemetry", expanded=False):
    fig_raw = go.Figure()
    fig_raw.add_trace(
        go.Scatter(
            x=asset_df["cycle"],
            y=asset_df["temp_c"],
            name="Core Temp (°C)",
            line=dict(color="#EF553B"),
        )
    )
    fig_raw.add_trace(
        go.Scatter(
            x=asset_df["cycle"],
            y=asset_df["vibration_mms"] * 15,
            name="Vibration (scaled x15)",
            line=dict(color="#00CC96"),
        )
    )
    fig_raw.add_trace(
        go.Scatter(
            x=asset_df["cycle"],
            y=asset_df["pressure_bar"] * 8,
            name="Pressure (scaled x8)",
            line=dict(color="#636EFA"),
        )
    )
    fig_raw.update_layout(
        title=f"Asset #{machine_id} Degradation Profile Across Lifecycle",
        xaxis_title="Operating Cycle",
        yaxis_title="Normalized Sensor Magnitude",
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig_raw, width="stretch")

st.markdown("---")

# Main Two-Column Modeling Workspace
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Darts Sensor Drift Forecast (Vibration mm/s)")
    st.write(
        "Compares a classical statistical baseline (**Exponential Smoothing**) against a deep learning architecture (**N-BEATS**)."
    )
    if st.button("Run Darts Forecast", type="primary"):
        with st.spinner("Training Exponential Smoothing & N-BEATS..."):
            res = run_cached_darts(df, machine_id=machine_id, horizon=horizon)

            m1, m2 = st.columns(2)
            m1.metric("ExpSmoothing MAPE", f"{res['stat_mape']}%")
            m2.metric("Darts N-BEATS MAPE", f"{res['nbeats_mape']}%")

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    y=res["actual"],
                    name="Actual Vibration",
                    line=dict(color="#FFFFFF", width=2.5),
                )
            )
            fig.add_trace(
                go.Scatter(
                    y=res["stat_forecast"],
                    name=f"ExpSmoothing ({res['stat_mape']}% MAPE)",
                    line=dict(color="#FFA15A", dash="dash"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    y=res["nbeats_forecast"],
                    name=f"Darts N-BEATS ({res['nbeats_mape']}% MAPE)",
                    line=dict(color="#00CC96", width=2.5),
                )
            )
            fig.update_layout(
                xaxis_title="Forecast Horizon Step",
                yaxis_title="Vibration (mm/s)",
                height=380,
                margin=dict(l=20, r=20, t=30, b=20),
            )
            st.plotly_chart(fig, width="stretch")

with col2:
    st.subheader("2. PyTorch LSTM Remaining Useful Life (RUL)")
    st.write(
        "Extracts rolling window features via **sktime** and predicts remaining cycles before failure using a custom **PyTorch LSTM**."
    )
    if st.button("Train & Predict RUL", type="primary"):
        with st.spinner("Extracting sktime features & running PyTorch LSTM..."):
            model = get_trained_lstm(df, test_machine_id=machine_id)

            X_test, y_test, _ = create_sequences(
                df[df["machine_id"] == machine_id], seq_length=20
            )
            model.eval()
            with torch.no_grad():
                preds = model(torch.tensor(X_test)).numpy()

            mae = float(abs(y_test - preds).mean())
            st.metric("Test Asset Mean Absolute Error (MAE)", f"{mae:.2f} cycles")

            fig2 = go.Figure()
            fig2.add_trace(
                go.Scatter(
                    y=y_test,
                    name="True RUL (Cycles)",
                    line=dict(color="#636EFA", width=2.5),
                )
            )
            fig2.add_trace(
                go.Scatter(
                    y=preds,
                    name="PyTorch Predicted RUL",
                    line=dict(color="#EF553B", width=2.5),
                )
            )
            fig2.update_layout(
                xaxis_title="Sliding Window Sequence Index",
                yaxis_title="Remaining Useful Life (Cycles)",
                height=380,
                margin=dict(l=20, r=20, t=30, b=20),
            )
            st.plotly_chart(fig2, width="stretch")
