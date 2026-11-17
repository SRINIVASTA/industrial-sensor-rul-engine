import streamlit as st
import torch
import plotly.graph_objects as go
from src.data_simulator import generate_sensor_data
from src.preprocessing import extract_sktime_features, create_sequences
from src.forecasting_darts import forecast_sensor_drift
from src.rul_pytorch import train_rul_model

st.set_page_config(page_title="Industrial Sensor RUL & Drift Engine", layout="wide")
st.title("⚙️ Industrial Predictive Maintenance & Sensor Drift Engine")
st.caption("Powered by PyTorch (LSTM) • Darts (N-BEATS & Exponential Smoothing) • sktime")

df = generate_sensor_data(n_machines=15)
machine_id = st.sidebar.selectbox("Select Turbine Asset ID", df["machine_id"].unique())

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Darts Sensor Drift Forecast (Vibration mm/s)")
    if st.button("Run Darts Forecast"):
        with st.spinner("Training Exponential Smoothing & N-BEATS..."):
            res = forecast_sensor_drift(df, machine_id=machine_id, horizon=15)
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=res["actual"], name="Actual Sensor", line=dict(color="white")))
            fig.add_trace(go.Scatter(y=res["stat_forecast"], name=f"ExpSmoothing (MAPE: {res['stat_mape']}%)", line=dict(dash="dash")))
            fig.add_trace(go.Scatter(y=res["nbeats_forecast"], name=f"Darts N-BEATS (MAPE: {res['nbeats_mape']}%)", line=dict(color="#00CC96")))
            st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("2. PyTorch LSTM Remaining Useful Life (RUL)")
    if st.button("Train & Predict RUL"):
        with st.spinner("Extracting sktime features & training PyTorch LSTM..."):
            _ = extract_sktime_features(df)
            X, y, _ = create_sequences(df[df["machine_id"] != machine_id], seq_length=20)
            model = train_rul_model(X, y, epochs=20)
            
            X_test, y_test, _ = create_sequences(df[df["machine_id"] == machine_id], seq_length=20)
            model.eval()
            with torch.no_grad():
                preds = model(torch.tensor(X_test)).numpy()
            
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(y=y_test, name="True RUL (Cycles)"))
            fig2.add_trace(go.Scatter(y=preds, name="PyTorch Predicted RUL"))
            st.plotly_chart(fig2, use_container_width=True)
