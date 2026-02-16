import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

data = pd.read_csv('data/training_data.csv')

X = data[['distance', 'speed', 'hour', 'weekday']]
Y = data['eta']

model = RandomForestRegressor(n_estimators=100)
model.fit(X,Y)

joblib.dump(model,"models/eta_model.pkl")

print("Model trained and saved to 'models/eta_model.pkl'")