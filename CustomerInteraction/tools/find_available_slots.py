import csv
import os
def get_available_slots(center_id, date):
    ALL_SLOTS = ["12:00", "13:00", "14:00", "15:00"]

    occupied = set()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CSV_PATH = os.path.join(BASE_DIR, "data", "occupied_slots.csv")
    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["center_id"] == center_id and row["date"] == date:
                occupied.add(row["time"])

    available = [s for s in ALL_SLOTS if s not in occupied]

    return available
