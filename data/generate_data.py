import pandas as pd 
import numpy as np 

rows = 5000
data = []

for _ in range(rows):
    distance = np.random.uniform(0.5,10) 
    speed = np.random.uniform(10,40)
    hour = np.random.randint(0,24)
    weekday = np.random.randint(0,7)

    traffic_factor = 1.5 if 8 <= hour <= 10 or 17 <= hour <= 19 else 1.0
    eta = (distance / speed) * traffic_factor
    data.append([distance, speed, hour, weekday, eta])

df = pd.DataFrame(data, columns=['distance', 'speed', 'hour', 'weekday', 'eta'])
df.to_csv('data/training_data.csv', index=False)
print("Data generated and saved to 'data/training_data.csv'")

