from .node_base import Node
from ..diagnosis_engine import diagnose

class DiagnosisAgentNode(Node):
    def __init__(self,name="DiagnosisAgent"):
        super().__init__(name)
    def run(self, input_data):
        sensor_row = input_data.get("sensor_data", [])
        latent = input_data["latent_vector"]
        recon_error = input_data["reconstruction_error"]

        diagnosis = diagnose(
            sensor_row=sensor_row,
            latent_vector=latent,
            recon_error=recon_error
        )

        return {
            **input_data,
            "diagnosis": diagnosis
        }
