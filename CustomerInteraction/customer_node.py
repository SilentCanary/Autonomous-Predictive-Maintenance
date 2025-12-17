from .node_base import Node
import asyncio
from .agent3 import process_user_turn,prepare_initial_conversation

class customer_node(Node):
    def __init__(self, name="CustomerAgent"):
        super().__init__(name)

    async def _run_async(self, input_data):
        memory = input_data.get("memory", [])
        user_message = input_data.get("user_message")

        # FIRST TURN
        if not memory:
            diagnosis_json = {
                "diagnosis": input_data["diagnosis"],
                "confidence": input_data["confidence"],
                "explanation": input_data["explanation"]
            }

            result = await prepare_initial_conversation(diagnosis_json)

            return {
                **input_data,
                "memory": result["memory"],
                "agent_message": result["agent_message"]
            }

        # SUBSEQUENT TURNS
        result = await process_user_turn(memory, user_message)

        return {
            **input_data,
            "memory": result["memory"],
            "agent_message": result["agent_message"],
            "tool_used": result.get("tool_used"),
            "tool_result": result.get("tool_result")
        }
    
    def run(self,input_data):
        return asyncio.run(self._run_async(input_data))