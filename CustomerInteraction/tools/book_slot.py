import csv
import os
def book_slot(center_id, date, time):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CSV_PATH = os.path.join(BASE_DIR, "data", "occupied_slots.csv")
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([center_id, date, time])

    return {
        "status": "CONFIRMED",
        "center_id": center_id,
        "date": date,
        "time": time
    }
