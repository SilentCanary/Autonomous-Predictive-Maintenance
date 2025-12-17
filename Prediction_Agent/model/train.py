# train_predictive_model.py
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from xgboost import XGBClassifier
import joblib
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("../dataset/sensor_data_simulated.csv")
y = df["failed"].values
X_raw = df.drop("failed", axis=1).values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

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

input_dim = X_scaled.shape[1]
ae = SensorAutoencoder(input_dim=input_dim, latent_dim=32)
ae.train()

normal_mask = y == 0
X_train_ae = torch.tensor(X_scaled[normal_mask], dtype=torch.float32)
dataset = TensorDataset(X_train_ae)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

optimizer = torch.optim.Adam(ae.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
epochs = 30

for epoch in range(epochs):
    total_loss = 0
    for (batch,) in loader:
        recon, _ = ae(batch)
        loss = loss_fn(recon, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.6f}")

torch.save(ae.state_dict(), "models/autoencoder.pt")
print("✅ Autoencoder trained & saved.")

ae.eval()
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
with torch.no_grad():
    recon, z = ae(X_tensor)
    recon_error = ((recon - X_tensor) ** 2).mean(dim=1).unsqueeze(1)

X_features = torch.cat([z, recon_error], dim=1).numpy()

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss"
)
xgb_model.fit(X_features, y)
joblib.dump(xgb_model, "models/xgb_model.pkl")
print("✅ XGBoost trained & saved.")


y_pred = xgb_model.predict(X_features)
report = classification_report(y, y_pred, target_names=["Normal", "Failed"])
print("\n=== Classification Report ===")
print(report)

with open("classification_report.txt", "w") as f:
    f.write(report)

joblib.dump(scaler, "models/scaler.pkl")
print("✅ All done. Models + report saved.")
