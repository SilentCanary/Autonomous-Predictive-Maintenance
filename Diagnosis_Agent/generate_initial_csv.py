import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import joblib
import os

# ===================== Paths =====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # PREDICTIVE/
MODEL_DIR = os.path.join(BASE_DIR, "Prediction_Agent", "model", "models")

INPUT_CSV = "../Prediction_Agent/dataset/sensor_data_simulated.csv"
OUTPUT_CSV = "../Prediction_Agent/dataset/diagnosis_latent_dataset.csv"

# ===================== Load scaler =====================
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

# ===================== Autoencoder =====================
class SensorAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

# ===================== Load data =====================
df = pd.read_csv(INPUT_CSV)

sensor_cols = [
    "engine_temp",
    "vibration",
    "oil_pressure",
    "rpm",
    "battery_voltage"
]

X_raw = df[sensor_cols].values
X_scaled = scaler.transform(X_raw)

# ===================== Load AE =====================
input_dim = X_scaled.shape[1]
ae = SensorAutoencoder(input_dim=input_dim, latent_dim=32)
ae.load_state_dict(torch.load(os.path.join(MODEL_DIR, "autoencoder.pt")))
ae.eval()

# ===================== Diagnosis Rules =====================
def enhanced_diagnosis(row, recon_error, recon_threshold):
    temp = row["engine_temp"]
    vib  = row["vibration"]
    oil  = row["oil_pressure"]
    rpm  = row["rpm"]
    volt = row["battery_voltage"]

    if temp > 105 and vib > 7:
        return "overheating", "Engine overheating: extreme temperature (>105°C) with high vibration"

    if vib > 9 or (vib > 6 and abs(rpm - 1200) > 300):
        return "bearing_fault", "Bearing fault: excessive vibration or vibration with RPM instability"

    if oil < 25 and temp > 95:
        return "low_oil", "Low oil pressure: pressure drop combined with rising temperature"

    if volt < 11.8 and rpm < 800:
        return "electrical_issue", "Electrical issue: low voltage affecting engine RPM"

    if rpm > 2200 or (rpm > 2000 and vib > 5):
        return "rpm_anomaly", "RPM anomaly: sustained high RPM with mechanical vibration"

    if recon_error > recon_threshold:
        return "unknown_anomaly", "Unrecognized anomaly pattern detected by autoencoder"

    return "normal", "Normal operation: all sensor values within safe operating ranges"

# ===================== Run AE =====================
with torch.no_grad():
    tensor = torch.tensor(X_scaled, dtype=torch.float32)
    recon, z = ae(tensor)
    recon_error = ((recon - tensor) ** 2).mean(dim=1).numpy()

# ===================== Threshold =====================
recon_threshold = np.percentile(recon_error, 95)

# ===================== Build Dataset =====================
records = []

for i in range(len(df)):
    latent_vector = z[i].numpy()

    diag_label, diag_text = enhanced_diagnosis(
        df.iloc[i],
        recon_error[i],
        recon_threshold
    )

    record = {f"z{j+1}": latent_vector[j] for j in range(len(latent_vector))}

    # Save raw sensors for explainability
    for col in sensor_cols:
        record[col] = df.iloc[i][col]

    record["reconstruction_error"] = recon_error[i]
    record["is_anomalous"] = int(recon_error[i] > recon_threshold)
    record["diagnosis"] = diag_label
    record["diagnosis_text"] = diag_text

    records.append(record)

latent_df = pd.DataFrame(records)

# ===================== Severity (meaningful bins) =====================
latent_df["severity"] = pd.cut(
    latent_df["reconstruction_error"],
    bins=[0, recon_threshold, recon_threshold * 1.5, recon_threshold * 2, np.inf],
    labels=["normal", "medium", "high", "critical"]
)

# ===================== Save =====================
latent_df.to_csv(OUTPUT_CSV, index=False)

# ===================== Report =====================
print(f"\n✅ Diagnosis latent dataset saved to {OUTPUT_CSV}")
print("\nDiagnosis distribution:")
print(latent_df["diagnosis"].value_counts())

print(f"\nAnomaly threshold: {recon_threshold:.6f}")
print(f"Anomalous cases: {latent_df['is_anomalous'].sum()} "
      f"({100 * latent_df['is_anomalous'].mean():.1f}%)")

print("\nSample records:")
print(
    latent_df[
        ["diagnosis", "severity", "reconstruction_error", "is_anomalous", "diagnosis_text"]
    ].head(10)
)
