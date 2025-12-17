import csv
import math

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371 

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

#for prototyping later to replace with actual api 
CITY_COORDS = {
    "shamli": (29.4500, 77.3100),
    "muzaffarnagar": (29.4727, 77.7085),
    "meerut": (28.9845, 77.7064)
}


def find_nearest_service_center(city):
    city_key=city.lower()
    if city_key not in CITY_COORDS:
        return None
    user_lat,user_long=CITY_COORDS[city_key]
    import os
    import pandas as pd

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CSV_PATH = os.path.join(BASE_DIR, "data", "service_centers.csv")

    with open(CSV_PATH,newline="") as f:
        reader=csv.DictReader(f)
        centers=list(reader)
    for c in centers:
        if c["city"].lower()==city.lower():
            return{
                "center_id":c["center_id"],
                "name":c["name"],
                "city":c["city"]
            }
    nearest = None
    min_distance = float("inf")

    for c in centers:
        dist = haversine_distance(
            user_lat,
            user_long,
            float(c["latitude"]),
            float(c["longitude"])
        )

        if dist < min_distance:
            min_distance = dist
            nearest = c

    return {
        "center_id": nearest["center_id"],
        "name": nearest["name"],
        "city": nearest["city"],
        "distance_km": round(min_distance, 2)
    }