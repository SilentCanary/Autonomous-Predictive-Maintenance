import random
import time

class SensorSimulator:
    def __init__(self):
        self.engine_temp = 85
        self.rpm = 1800
        self.speed = 40
        self.vibration = 0.3
        self.coolant = 90

    def step(self):
        # Normal fluctuations
        self.engine_temp += random.uniform(-0.5, 0.8)
        self.rpm += random.randint(-50, 50)
        self.speed += random.randint(-2, 2)
        self.vibration += random.uniform(-0.02, 0.05)
        self.coolant -= random.uniform(0.01, 0.05)

        # Clamp values
        self.engine_temp = max(60, min(120, self.engine_temp))
        self.coolant = max(0, min(100, self.coolant))
        if random.random() < 0.08:  # 8% chance
            print("\n⚠️  SENSOR SPIKE DETECTED")
            self.engine_temp += random.uniform(10, 25)
            self.vibration += random.uniform(0.5, 1.0)
        return [
            round(self.engine_temp, 2),
            self.rpm,
            self.speed,
            round(self.vibration, 3),
            round(self.coolant, 2)
        ]
