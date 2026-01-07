from .node_base import Node
import asyncio
from .agent import customer_conversation_loop, convert_to_human_explainable

class CustomerAgentNode(Node):
    def __init__(self, name="CustomerAgent"):
        super().__init__(name)

    async def _run_async(self, input_data):
        diagnosis = {
            "diagnosis": input_data.get("diagnosis", "unknown"),
            "confidence": input_data.get("confidence", 1.0),
            "explanation": input_data.get("explanation", "")
        }

        # Convert to human-readable text
        human_text = convert_to_human_explainable(diagnosis)

        # Ensure memory is always a list
        conversational_memory = input_data.get("memory")
        if not isinstance(conversational_memory, list):
            conversational_memory = []

        initial_agent_text = (
            "==============================\n"
            "🚗 Vehicle Customer Care Agent\n"
            "==============================\n\n"
            f"{human_text}\n\n"
            "Aap service ke liye kab aa sakte hai?"
        )
        conversational_memory.append({"role": "agent", "text": initial_agent_text})
        print(initial_agent_text)

        # Run the conversation loop
        await customer_conversation_loop(conversational_memory)

        # Return final state
        return {
            **input_data,
            "memory": conversational_memory,
            "agent_message": conversational_memory[-1]["text"] if conversational_memory else None
        }

    def run(self, input_data):
        return asyncio.run(self._run_async(input_data))
