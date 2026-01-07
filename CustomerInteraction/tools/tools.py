from langchain.tools import tool
from .find_available_slots import get_available_slots
from .book_slot import book_slot
from .find_nearest import find_nearest_service_center
from .normalise_date import normalize_datetime


@tool
def find_center(city: str) -> dict:
    """
    Find the nearest service center given a city.
    Returns center_id and center details.
    """
    return find_nearest_service_center(city)


@tool
def get_slot(center_id: str, date: str) -> list:
    """
    Get available slots for a given center_id and date.
    """
    return get_available_slots(center_id, date)


@tool
def book_slot_tool(center_id: str, date: str, time: str) -> dict:
    """
    Book a service slot using center_id, date, and time.
    """
    return book_slot(center_id, date, time)


@tool
def normalise_date(date: str) -> dict:
    """
    Normalize relative or ambiguous date/time expressions
    like kal, parso, shaam into exact date/time.
    """
    return normalize_datetime(date)
