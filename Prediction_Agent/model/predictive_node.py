from typing import Any, Dict
from .node_base import Node
from .prediction_tool import predict_failure 

class PredictiveAgentNode(Node):
    def __init__(self,name="PredictiveAgent"):
        super().__init__(name)

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        sensor_data = input_data.get("sensor_data", [])
        if not sensor_data:
            return {"error": "No sensor data provided"}

        result = predict_failure(sensor_data)
        return result
    def should_run_next(self, output):
    # Only run Diagnosis if failure probability is above threshold
        return output.get("failure_probability", 0) > 0.2

