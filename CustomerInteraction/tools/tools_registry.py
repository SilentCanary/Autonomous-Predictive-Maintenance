# tools/tool_registry.py
from .tools import (
    find_center,
    get_slot,
    book_slot_tool,
    normalise_date
)

TOOL_REGISTRY = {
    "find_center": find_center,
    "get_slot": get_slot,
    "book_slot": book_slot_tool,
    "normalise_date": normalise_date
}
