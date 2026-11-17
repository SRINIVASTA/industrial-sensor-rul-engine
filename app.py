import time
import numpy as np
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
    "Real-Time Telemetry Stream & Applied ML Pipeline • PyTorch (LSTM) • Darts (N-BEATS) • sktime"
)

if "darts_active" not in st.session_state:
    st.session_state.darts_active = False
if "rul_active" not in st.session_state:
    st.session_state.rul_active = False
if "sim_seed" not in st.session_state:
    st.session_state.sim_seed = 42


@st.cache_data
def load_telemetry_data(seed: int):
    return generate_sensor_data(n_machines=15, seed=seed)


@st.cache_data(max_entries=3)
def run_cached_darts(df, machine_id: int, horizon: int = 15):
    return forecast_sensor_drift(df, machine_id=machine_id, horizon=horizon)


@st.cache_resource(max_entries=3)
def get_trained_lstm(df, test_machine_id: int):
    _ = extract_sktime_features(df[df["machine_id"] <= 3])
    X_train, y_train, _ = create_sequences(
        df[df["machine_id"] != test_machine_id], seq_length=20
    )
    return train_rul_model(X_train, y_train, epochs=20)


# Sidebar Controls
st.sidebar.header("🎛️ Asset & Stream Controls")
machine_id = st.sidebar.selectbox(
    "Select Turbine Asset ID", list(range(1, 16))
)
horizon = st.sidebar.slider("Darts Forecast Horizon (Cycles)", 10, 25, 15)

if st.sidebar.button("🔄 Generate New Synthetic Fleet"):
    st.session_state.sim_seed = int(np.random.randint(1, 10000))
    st.rerun()

st.sidebar.markdown("---")
stream_live = st.sidebar.toggle("🔴 Enable Live Sensor Telemetry Stream", value=False)
stream_speed = st.sidebar.slider("Stream Refresh Rate (sec)", 0.1, 1.0, 0.3)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Tech Stack:**\n"
    "- `sktime`: Rolling window feature extraction\n"
    "- `Darts`: Exponential Smoothing vs. N-BEATS\n"
    "- `PyTorch`: 2-Layer LSTM with Huber Loss\n"
    "- `Plotly` & `Streamlit`: Real-time diagnostics"
)

df = load_telemetry_data(st.session_state.sim_seed)
asset_df = df[df["machine_id"] == machine_id].sort_values("cycle").reset_index(drop=True)

# ---------------------------------------------------------
# SECTION 1: LIVE TELEMETRY STREAMING SIMULATOR
# ---------------------------------------------------------
st.subheader(f"📡 Live Sensor Telemetry Feed — Turbine Asset #{machine_id}")

if stream_live:
    with st.spinner("Initializing PyTorch LSTM inference engine..."):
        live_model = get_trained_lstm(df, test_machine_id=machine_id)
        X_seq, y_seq, _ = create_sequences(asset_df, seq_length=20)
        live_model.eval()
        with torch.no_grad():
            live_rul_preds = live_model(torch.tensor(X_seq)).numpy()

    kpi_placeholder = st.empty()
    chart_placeholder = st.empty()

    # Stream the final 45 critical cycles so rapid degradation is visible immediately
    start_idx = max(25, len(asset_df) - 45)
    for idx in range(start_idx, len(asset_df)):
        current_slice = asset_df.iloc[: idx + 1]
        latest = current_slice.iloc[-1]
        prev = current_slice.iloc[-2]

        pred_idx = min(idx - 20, len(live_rul_preds) - 1)
        est_rul = max(0.0, float(live_rul_preds[pred_idx]))

        with kpi_placeholder.container():
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Live Cycle", f"#{int(latest['cycle'])}", delta="+1 cycle")
            m2.metric(
                "Vibration",
                f"{latest['vibration_mms']:.2f} mm/s",
                delta=f"{latest['vibration_mms'] - prev['vibration_mms']:+.2f}",
                delta_color="inverse",
            )
            m3.metric(
                "Core Temp",
                f"{latest['temp_c']:.1f} °C",
                delta=f"{latest['temp_c'] - prev['temp_c']:+.1f}",
                delta_color="inverse",
            )
            m4.metric(
                "Pressure",
                f"{latest['pressure_bar']:.2f} bar",
                delta=f"{latest['pressure_bar'] - prev['pressure_bar']:+.2f}",
            )
            status_label = "CRITICAL" if est_rul < 25 else "NORMAL"
            m5.metric(
                f"PyTorch Est. RUL ({status_label})",
                f"{est_rul:.1f} cycles",
                delta=f"True RUL: {int(latest['RUL'])}",
            )

        with chart_placeholder.container():
            plot_df = current_slice[["cycle", "vibration_mms", "pressure_bar"]].set_index("cycle")
            st.line_chart(plot_df, height=260)

        time.sleep(stream_speed)
else:
    st.info(
        "👈 Toggle **'Enable Live Sensor Telemetry Stream'** in the left sidebar to watch live sensor packets and PyTorch RUL predictions update in real time."
    )
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Lifecycle", f"{int(asset_df['cycle'].max())} cycles")
    k2.metric("Peak Vibration", f"{asset_df['vibration_mms'].max():.2f} mm/s")
    k3.metric("Peak Core Temp", f"{asset_df['temp_c'].max():.1f} °C")
    k4.metric("Final Pressure", f"{asset_df['pressure_bar'].iloc[-1]:.2f} bar")

st.markdown("---")

# ---------------------------------------------------------
# SECTION 2: BATCH MODELING & BENCHMARKING
# ---------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Darts Sensor Drift Forecast (Vibration mm/s)")
    st.write(
        "Compares classical **Exponential Smoothing** against deep **N-BEATS**."
    )
    if st.button("Run Darts Forecast", type="primary"):
        st.session_state.darts_active = True

    if st.session_state.darts_active:
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
            height=360,
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig, width="stretch", key="darts_batch_chart")

with col2:
    st.subheader("2. PyTorch LSTM Remaining Useful Life (RUL)")
    st.write(
        "Extracts rolling window features via **sktime** and predicts remaining cycles using **PyTorch LSTM**."
    )
    if st.button("Train & Predict RUL", type="primary"):
        st.session_state.rul_active = True

    if st.session_state.rul_active:
        with st.spinner("Extracting sktime features & running PyTorch LSTM..."):
            model = get_trained_lstm(df, test_machine_id=machine_id)

        X_test, y_test, _ = create_sequences(asset_df, seq_length=20)
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
            height=360,
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig2, width="stretch", key="rul_batch_chart")
