from typing import List, Dict, Any
import os
import torch
import torch.nn as nn
import joblib
import numpy as np
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("predictive_agent")

MODEL_DIR = "models"
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

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

ae_input_dim = scaler.mean_.shape[0]
ae = SensorAutoencoder(input_dim=ae_input_dim, latent_dim=32)
ae.load_state_dict(torch.load(os.path.join(MODEL_DIR, "autoencoder.pt")))
ae.eval()

xgb_model = joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))

def predict_failure(sensor_data: List[float]) -> Dict[str, Any]:
    sensor_array = np.array(sensor_data)
    if len(sensor_array.shape) > 1:
        sensor_array = sensor_array.flatten()[np.newaxis, :]
    else:
        sensor_array = sensor_array.reshape(1, -1)

    scaled = scaler.transform(sensor_array)

    with torch.no_grad():
        tensor = torch.tensor(scaled, dtype=torch.float32)
        recon, latent = ae(tensor)
        recon_error = ((tensor - recon) ** 2).mean(dim=1).item()
        latent_vector = latent.numpy().flatten().tolist()

    features = np.hstack([latent.numpy(), [[recon_error]]])
    failure_prob = xgb_model.predict_proba(features)[0][1]

    return {
        "failure_probability": float(failure_prob),
        "latent_vector": latent_vector,
        "reconstruction_error": float(recon_error)
    }

@mcp.tool()
async def predict(sensor_data: List[float]) -> Dict[str, Any]:
    return predict_failure(sensor_data)

def main():
    mcp.run(transport='stdio')

if __name__ == "__main__":
    # Example test without MCP
    example_data = [71.03, 1.25, 73.46, 1879.78, 12.52]
    print(predict_failure(example_data))

    main()
