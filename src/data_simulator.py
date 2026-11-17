import numpy as np
import pandas as pd

def generate_sensor_data(n_machines=15, max_cycles=200, seed=42):
    """Simulates multivariate industrial sensor telemetry until failure."""
    np.random.seed(seed)
    records = []

    for machine_id in range(1, n_machines + 1):
        life_span = np.random.randint(130, max_cycles)
        for cycle in range(1, life_span + 1):
            # Exponential degradation factor as machine approaches failure
            wear = np.exp((cycle / life_span) * 3.5) / np.exp(3.5)
            
            temp_c = 65.0 + (28.0 * wear) + np.random.normal(0, 0.8)
            vibration_mms = 1.2 + (4.5 * wear) + np.random.normal(0, 0.15)
            pressure_bar = 10.5 - (2.1 * wear) + np.random.normal(0, 0.1)
            rpm = 3000 - (180 * wear) + np.random.normal(0, 12.0)
            rul = life_span - cycle

            records.append({
                "machine_id": machine_id,
                "cycle": cycle,
                "temp_c": round(temp_c, 3),
                "vibration_mms": round(vibration_mms, 3),
                "pressure_bar": round(pressure_bar, 3),
                "rpm": round(rpm, 2),
                "RUL": rul
            })

    df = pd.DataFrame(records)
    return df

if __name__ == "__main__":
    df = generate_sensor_data()
    df.to_csv("data/simulated_turbine_telemetry.csv", index=False)
    print(f"Generated {len(df)} telemetry rows across {df['machine_id'].nunique()} machines.")
