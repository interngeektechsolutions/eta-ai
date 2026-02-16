from fastapi import FastAPI
import joblib


app = FastAPI()

model = joblib.load("models/eta_model.pkl")

@app.post("/predict_eta")
def predict_eta(data:dict):
    features = [[
        data['distance'],
        data['speed'],
        data['hour'],
        data['weekday']
    ]]
    eta = model.predict(features)[0]
    return {"estimated_time_of_arrival": round(float(eta),2)}