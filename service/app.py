from fastapi import FastAPI, HTTPException
from pydantic import BaseModel,Field
from contextlib import asynccontextmanager
import logging
import os
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "models/eta_model.pkl")
ml_model = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        ml_model["eta"] = joblib.load(MODEL_PATH)
        logger.info(f"Model loaded from {MODEL_PATH}")
    except FileNotFoundError:
        logger.error(f"Model file not found at '{MODEL_PATH}'")
        raise RuntimeError(f"Model file not found at '{MODEL_PATH}'")
    yield

    ml_model.clear()
    logger.info("Model unloaded")


app = FastAPI(
    title = "ETA Prediction API",
    description = "Estimated Time of Arrival  prediction service",
    version="1.0.0",
    lifespan=lifespan
)

class ETARequest(BaseModel):
    distance: float = Field(...,gt=0,description="Distance in km")
    speed:    float = Field(...,gt=0,description="Speed in km/h")
    hour:    int = Field(...,ge=0,le=23,description="Hour of the day (0-23)")
    weekday:  int = Field(...,ge=0,le=6,description="Day of the week (0=Monday, 6=Sunday)")

class ETAResponse(BaseModel):
    estimated_time_of_arrival: float
    unit:str = "hours"
    
    
@app.get("/health")
def health_check():
    return {"status":"healthy","model_loaded":"eta" in ml_model}


@app.post("/predict_eta",response_model = ETAResponse)
def predict_eta(data:ETARequest):
    try:
        features = [[data.distance,data.speed,data.hour,data.weekday]]
        eta = ml_model["eta"].predict(features)[0]
        logger.info(f"Prediction:{eta:.4f} for input {data.model_dump()}")
        return ETAResponse(estimated_time_of_arrival=round(float(eta),2))
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500,detail="Prediction failed")
    
    