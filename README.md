# ⚙️ Industrial Sensor RUL & Drift Engine

[![Localhost](https://shields.io)](http://localhost:8501)

A local web application (`http://localhost:8501`) for **real-time industrial sensor telemetry streaming**, **sensor drift forecasting**, and **Remaining Useful Life (RUL) prediction** using **PyTorch**, **Darts**, **sktime**, and **Streamlit**.

## About

Industrial Sensor RUL & Drift Engine provides a modular, local-first applied machine learning pipeline for predictive maintenance—built from scratch with statistical time-series models and custom neural networks rather than third-party LLM wrappers.

Built and maintained by [@SRINIVASTA](https://github.com), this repository hosts a real-time telemetry stream simulator, an `sktime` sliding-window feature extractor, a `Darts` statistical vs. deep forecasting benchmark (`ExponentialSmoothing` vs. `N-BEATS`), and a custom `PyTorch` 2-layer LSTM failure countdown engine.

## Modules in this repo

| Module | Description |
| :--- | :--- |
| `data_simulator.py` | Multivariate industrial turbine simulator generating correlated exponential mechanical wear curves across Temperature, Vibration, Pressure, and RPM until breakdown. |
| `preprocessing.py` | Time-series feature engineering pipeline using `sktime` (`WindowSummarizer`) for rolling lag/mean/std extraction and `MinMaxScaler` 3D sliding-window tensor generation. |
| `forecasting_darts.py` | Multi-step vibration drift forecasting benchmark comparing a classical statistical baseline (`ExponentialSmoothing`) against deep learning (`NBEATSModel`) via MAPE. |
| `rul_pytorch.py` | Custom 2-layer `PyTorch` LSTM architecture (`RULPredictorLSTM`) trained with `HuberLoss` to predict exact Remaining Useful Life (cycles) on unseen assets. |
| `app.py` | Interactive `Streamlit` and `Plotly` diagnostic dashboard (`http://localhost:8501`) featuring a real-time streaming sensor feed, cached model inference, and side-by-side performance benchmarks. |

### Architecture & Workflow

```mermaid
flowchart TD
    subgraph User_Interface_Layer["User Interface Layer - Localhost 8501"]
        A["app.py<br>Streamlit Diagnostic Gateway"] e1@-->|"Select Turbine Asset & Mode"| B{"Workflow Router"}
    end

    subgraph Telemetry_Simulation_Layer["Telemetry & Feature Engineering Layer"]
        B e2@-->|"Simulated Fleet Seed"| C["data_simulator.py<br>Multivariate Sensor Wear Generator"]
        C e3@-->|"Raw Telemetry Streams"| D["preprocessing.py<br>sktime WindowSummarizer & 3D Sequences"]
    end

    subgraph Live_Stream_Layer["Real-Time Telemetry Stream Mode"]
        D e4@-->|"Rolling 20-Cycle Windows"| E["Live Packet Streamer<br>Real-Time KPIs & Alert Status"]
        E e5@-->|"Continuous Inference"| F["PyTorch RUL Countdown<br>NORMAL vs CRITICAL Alerting"]
    end

    subgraph Batch_Modeling_Layer["Batch Forecasting & RUL Benchmark Layer"]
        D e6@-->|"Univariate Vibration Series"| G["forecasting_darts.py<br>ExpSmoothing vs Darts N-BEATS"]
        D e7@-->|"Multivariate 3D Tensors"| H["rul_pytorch.py<br>2-Layer PyTorch LSTM + HuberLoss"]
        G e8@-->|"MAPE Benchmark"| I["Plotly Drift Comparison Chart"]
        H e9@-->|"Cycle MAE Benchmark"| J["Plotly True vs Predicted RUL Chart"]
    end

    e1@{ animate: true }
    e2@{ animate: true }
    e3@{ animate: true }
    e4@{ animate: true }
    e5@{ animate: true }
    e6@{ animate: true }
    e7@{ animate: true }
    e8@{ animate: true }
    e9@{ animate: true }
```

### Key Capabilities

* **Real-Time Sensor Stream Simulator:** Replays multivariate turbine degradation cycles in real time with live delta KPIs, dynamic charts, and automated `NORMAL` / `CRITICAL` health status alerts—requiring zero physical IoT hardware.
* **Statistical vs. Deep Forecasting (`Darts`):** Benchmarks classical `ExponentialSmoothing` (`6.95% MAPE`) against deep `N-BEATS` (`3.53% MAPE`) on multi-step vibration drift forecasting.
* **Custom Failure Countdown (`PyTorch` + `sktime`):** Extracts rolling time-series statistics via `sktime` and trains a custom 2-layer `PyTorch` LSTM (`6.98 cycles MAE` on unseen test turbines).

## Repository Structure

```text
industrial-sensor-rul-engine/
├── data/
│   └── simulated_turbine_telemetry.csv   # Generated or NASA C-MAPSS sensor logs
├── src/
│   ├── __init__.py
│   ├── data_simulator.py                 # Multivariate exponential mechanical wear generator
│   ├── preprocessing.py                  # sktime WindowSummarizer & 3D sequence builder
│   ├── forecasting_darts.py              # Darts ExponentialSmoothing vs N-BEATS pipeline
│   └── rul_pytorch.py                    # Custom PyTorch LSTM architecture & training loop
├── app.py                                # Interactive Streamlit & Plotly diagnostic dashboard
├── requirements.txt                      # Python dependencies
└── README.md
```

## Installation & Local Setup

You can clone and run the Industrial Sensor RUL & Drift Engine locally across Windows, macOS, and Linux:

### 1. Clone the repository

```bash
git clone https://github.com/industrial-sensor-rul-engine.git
cd industrial-sensor-rul-engine
```

### 2. Set up Python environment

```bash
# Create virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS / Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Generate telemetry dataset (Optional)

```bash
python src/data_simulator.py
```

### 4. Launch the application

```bash
streamlit run app.py
```

Open your browser and navigate to `http://localhost:8501`.

## License

This repository is licensed under the MIT License — Copyright (c) 2026 [SRINIVASTA](https://github.com).

## Disclaimer

This project uses synthetically simulated multivariate industrial turbine telemetry for predictive maintenance modeling, benchmarking, and educational demonstration. For production deployment on physical machinery, models should be calibrated and validated against site-specific SCADA/PLC sensor baselines.

## ⭐ Support & Contribution

If you clone, test, or build upon this repository, please consider giving it a **Star (⭐)** on GitHub—it helps others discover the project and supports continued development.

* **Found a bug or have an idea?** Open an [Issue](https://github.com/industrial-sensor-rul-engine/issues).
* **Want to improve the code?** Feel free to **Fork** this repository and submit a Pull Request.

---

* **Lead Architect & Developer:** [Srinivasta](https://github.com)

### 🌐 Let’s Connect

[![LinkedIn](https://shields.io)](https://linkedin.com)
[![Kaggle](https://shields.io)](https://kaggle.com)
[![Email](https://shields.io)](mailto:tasrinivass@gmail.com)
[![GitHub](https://shields.io)](https://github.com)
[![Website](https://shields.io)](https://github.io)
