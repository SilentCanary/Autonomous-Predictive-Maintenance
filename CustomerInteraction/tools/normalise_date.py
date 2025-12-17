TIME_MAP = {
    "subah": ("09:00", "11:00"),
    "dopahar": ("12:00", "15:00"),
    "shaam": ("17:00", "19:00"),
    "raat": ("20:00", "22:00")
}

DAY_OFFSET = {
    "aaj": 0,
    "kal": 1,
    "parso": 2
}

#will need to later reformulate using llm so that we can handle incorrect spelling mistakes etc..

from datetime import datetime, timedelta
import re

def normalize_datetime(text: str):
    text = text.lower()

    today = datetime.today().date()

    day_offset = None
    for word, offset in {
        "aaj": 0,
        "kal": 1,
        "parso": 2
    }.items():
        if word in text:
            day_offset = offset
            break

    if day_offset is None:
        return {
            "error": "date_not_found",
            "message": "Date samajh nahi aayi"
        }

    date = today + timedelta(days=day_offset)

    exact_time = re.search(r"\b(\d{1,2})(?:\s*baje)?\b", text)

    if exact_time:
        hour = int(exact_time.group(1))
        return {
            "date": str(date),
            "time": f"{hour:02d}:00",
            "confidence": "exact"
        }

    for key, (start, end) in {
        "subah": ("09:00", "11:00"),
        "dopahar": ("12:00", "15:00"),
        "shaam": ("17:00", "19:00"),
        "raat": ("20:00", "22:00")
    }.items():
        if key in text:
            return {
                "date": str(date),
                "time_window": f"{start}-{end}",
                "confidence": "approx"
            }

    return {
        "error": "time_not_found",
        "message": "Time samajh nahi aaya"
    }
