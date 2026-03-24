import logging
import time
import numpy as np
import joblib

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


from fastapi.security import APIKeyHeader
from fastapi import Depends
 
# 1. LOAD TRAINED MODEL
 

try:
    model = joblib.load("automotive_maintenance_model.pkl")
except Exception as e:
    raise RuntimeError(f"Model failed to load: {e}")

 
# 2. LOGGING CONFIGURATION
 

logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

# ==========================================================
# API KEY CONFIGURATION
# ==========================================================

API_KEY = "mysecretkey"
api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        logger.warning("Unauthorized access attempt detected")
        raise HTTPException(status_code=403, detail="Unauthorized")
    

 
# 3. INITIALIZE FASTAPI
 

app = FastAPI(
    title="Automotive Predictive Maintenance API",
    description="Production-grade ML service for vehicle engine failure prediction.",
    version="1.0.0"
)

 
# 4. GLOBAL EXCEPTION HANDLER
 

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled Error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": str(exc)
        }
    )

 
# 5. MIDDLEWARE FOR REQUEST LOGGING & TIMING
 

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    logger.info(f"Incoming Request: {request.method} {request.url}")

    response = await call_next(request)

    process_time = time.time() - start_time

    logger.info(
        f"Completed: {request.method} {request.url} | "
        f"Status: {response.status_code} | "
        f"Time: {round(process_time, 4)} sec"
    )

    return response

 
# 6. REQUEST & RESPONSE SCHEMAS
 


class SensorInput(BaseModel):
    air_temperature: float = Field(
        ..., json_schema_extra={"example": 300}
    )
    process_temperature: float = Field(
        ..., json_schema_extra={"example": 310}
    )
    rotational_speed: float = Field(
        ..., json_schema_extra={"example": 1500}
    )
    torque: float = Field(
        ..., json_schema_extra={"example": 40}
    )
    tool_wear: float = Field(
        ..., json_schema_extra={"example": 120}
    )
    machine_type: int = Field(
        ..., 
        description="0=L, 1=M, 2=H",
        json_schema_extra={"example": 1}
    )

class PredictionResponse(BaseModel):
    failure_probability: float
    risk_level: str
    recommendation: str


 
# 7. HEALTH CHECK ENDPOINT
 

@app.get("/")
def health_check():
    return {
        "status": "Automotive Predictive Maintenance API is running"
    }

 
# 8. PREDICTION ENDPOINT
 

@app.post("/predict", response_model=PredictionResponse)
def predict(data: SensorInput, api_key: str = Depends(verify_api_key)):

    try:
        # Convert input into numpy array
        features = np.array([[
            data.air_temperature,
            data.process_temperature,
            data.rotational_speed,
            data.torque,
            data.tool_wear,
            data.machine_type
        ]])

        # Predict
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        # Risk categorization
        if probability < 0.3:
            risk = "Low"
            recommendation = "Normal operation"
        elif probability < 0.7:
            risk = "Medium"
            recommendation = "Schedule inspection"
        else:
            risk = "High"
            recommendation = "Immediate maintenance required"

        logger.info(
            f"Prediction made | Probability: {probability} | Risk: {risk}"
        )

        return {
            "failure_probability": round(float(probability), 4),
            "risk_level": risk,
            "recommendation": recommendation
        }

    except Exception as e:
        logger.error(f"Prediction Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    


