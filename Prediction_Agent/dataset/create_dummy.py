import numpy as np
import pandas as pd

num_sensors = 5           
timesteps = 1000 
window_size = 1            
failure_prob = 0.05        
np.random.seed(42)

sensor_ranges = {
    'engine_temp': (70, 110),         
    'vibration': (0, 10),             
    'oil_pressure': (20, 100),    
    'rpm': (600, 2000),              
    'battery_voltage': (12, 14.5)   
}


data = np.zeros((timesteps, num_sensors))
for i, (sensor, (low, high)) in enumerate(sensor_ranges.items()):
    data[:, i] = np.random.uniform(low, high, size=timesteps) + np.random.normal(0, 0.5, timesteps)


failed = np.zeros(timesteps)
num_failures = int(timesteps * failure_prob)

failure_indices = np.random.choice(timesteps, num_failures, replace=False)
for idx in failure_indices:
    data[idx, 0] += np.random.uniform(10, 30)  # temp spike
    data[idx, 1] += np.random.uniform(5, 15)   # vibration spike
    data[idx, 2] += np.random.uniform(-20, -5) # pressure drop
    failed[idx] = 1


df = pd.DataFrame(data, columns=list(sensor_ranges.keys()))
df['failed'] = failed.astype(int)


df.to_csv('sensor_data_simulated.csv', index=False)
print("Simulated vehicle sensor dataset saved to 'sensor_data_simulated.csv'")
print(df.head(10))
