from typing import Any, Dict

class Node:
    def __init__(self, name: str):
        self.name = name

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("Each node must implement the run method")
