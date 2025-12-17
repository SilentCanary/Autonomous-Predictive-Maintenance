from typing import Any, Dict
from  Prediction_Agent.model.predictive_node import PredictiveAgentNode
from  Diagnosis_Agent.models.diagnosis_node import DiagnosisAgentNode
from  CustomerInteraction.customer_node import customer_node

#helper for printing 
def block(title: str, emoji: str = "🔹", width: int = 60):
    print("\n" + "═" * width)
    print(f"{emoji}  {title}".center(width))
    print("═" * width)

def sub_block(title: str, emoji: str = "➡️", width: int = 60):
    print("\n" + "─" * width)
    print(f"{emoji}  {title}")
    print("─" * width)

def pretty_dict(d: dict, indent: int = 2):
    for k, v in d.items():
        print(" " * indent + f"• {k}: {v}")



class Langraph:
    def __init__(self):
        # Initialize nodes
        self.nodes = {}
        self.register_node(PredictiveAgentNode("PredictiveAgent"))
        self.register_node(DiagnosisAgentNode("DiagnosisAgent"))
        self.register_node(customer_node("CustomerAgent"))
        # self.register_node(CustomerAgentNode("CustomerAgent"))
        # ... add other nodes later

        # Define simple connections (edges)
        # For now, just a dict of node_name -> next_node_names
        self.edges = {
            "PredictiveAgent": ["DiagnosisAgent"],  # currently no next node
            "DiagnosisAgent": ["CustomerAgent"],
        }

    def register_node(self, node):
        self.nodes[node.name] = node

    def run_node(self, node_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        if node_name not in self.nodes:
            raise ValueError(f"Node {node_name} not found in Langraph")
        node = self.nodes[node_name]
        output = node.run(input_data)
        if node_name=="PredictiveAgent" or node_name=="DiagnosisAgent":
            block(f"{node_name} OUTPUT", "🧠")
            pretty_dict(output)

        next_nodes_to_run = getattr(node, "next_nodes", self.edges.get(node_name, []))
        for next_node in next_nodes_to_run:
            if next_node=="DiagnosisAgent":
                if hasattr(node, "should_run_next") and not node.should_run_next(output):
                    continue
                sub_block("Routing → Diagnosis Agent", "🩺")
                input_for_next = {
                    "sensor_data": input_data.get("sensor_data"),
                    "latent_vector": output.get("latent_vector"),
                    "reconstruction_error": output.get("reconstruction_error")
                }
                output = self.run_node(next_node, input_for_next)
                print(output)
                print()
            elif next_node=="CustomerAgent":
                sub_block("Routing → Customer Interaction Agent", "💬")
                diag = output.get("diagnosis", {})
                input_for_next = {
                    "diagnosis": diag.get("diagnosis"),
                    "confidence": diag.get("confidence"),
                    "explanation": diag.get("explanation"),
                    "memory": input_data.get("memory"),       # for multi-turn
                    "user_message": input_data.get("user_message")
                }
                output = self.run_node(next_node, input_for_next)
                print("\n🤖 Agent:")
                print("  " + output["agent_message"])
                while True:
                    user = input("\n🧑 Owner: ")
                    if user.lower() == "exit":
                        break

                    loop_input = {
                        "diagnosis": diag.get("diagnosis"),
                        "confidence": diag.get("confidence"),
                        "explanation": diag.get("explanation"),
                        "memory": output.get("memory"),
                        "user_message": user
                    }

                    output = self.run_node(next_node, loop_input)
                    print("Agent:", output["agent_message"])

        return output

if __name__ == "__main__":
    lg = Langraph()
    from sensor_simulator import SensorSimulator
    import time

    sim = SensorSimulator()
    while True:
        sensor_data = sim.step()
        block("📡 LIVE VEHICLE TELEMETRY", "📡")
        print(f"  Engine Temp   : {sensor_data[0]} °C")
        print(f"  RPM           : {sensor_data[1]}")
        print(f"  Speed         : {sensor_data[2]} km/h")
        print(f"  Vibration     : {sensor_data[3]}")
        print(f"  Coolant Level : {sensor_data[4]} %")

        lg.run_node("PredictiveAgent", {
            "sensor_data": sensor_data
        })

        time.sleep(2)

    # Example input for Predictive Agent
    example_input = {
        "sensor_data": [103.55293359680516,15.82548355646633,43.08055035044674,1304.0633646686429,13.146403597173785]
    }
    result = lg.run_node("PredictiveAgent", example_input)
    print("Langraph Output:", result)
