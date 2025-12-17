def rule_based_diagnosis(sensor_row):
    temp = sensor_row[0] #engine temp
    vib  = sensor_row[1] #vibration
    oil  = sensor_row[2] #oil pressure
    rpm  = sensor_row[3] #rpm
    volt = sensor_row[4] #batter voltage

    if temp > 105 and vib > 7:
        return {
            "diagnosis": "overheating",
            "confidence": 0.95,
            "explanation": "Extreme temperature with high vibration"
        }

    if vib > 9:
        return {
            "diagnosis": "bearing_fault",
            "confidence": 0.9,
            "explanation": "Excessive vibration detected"
        }

    if oil < 25:
        return {
            "diagnosis": "low_oil",
            "confidence": 0.85,
            "explanation": "Oil pressure below safe threshold"
        }

    return None  # Escalate
