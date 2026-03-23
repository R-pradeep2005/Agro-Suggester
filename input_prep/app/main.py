import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import random

app = FastAPI(title="Agro-Suggester Input Preparation Service")

class GeolocationInput(BaseModel):
    lat: float
    lon: float
    previous_crop: str
    previous_fertilizer_n: float
    previous_fertilizer_p: float
    previous_fertilizer_k: float

@app.get("/health")
async def health():
    return {"status": "ok", "service": "input_prep"}

# --- External API Mock Integrations ---

@app.get("/api/external/soil")
async def get_soil_data(lat: float, lon: float):
    """Mocks SoilGrids API spatial query response"""
    return {
        "ph": round(random.uniform(5.5, 8.0), 2),
        "organic_carbon": round(random.uniform(10, 50), 2),
        "nitrogen": round(random.uniform(10, 30), 2),
        "cec": round(random.uniform(15, 30), 2),
        "bdod": round(random.uniform(1.1, 1.5), 2),
        "sand": round(random.uniform(20, 40), 1),
        "silt": round(random.uniform(20, 40), 1),
        "clay": round(random.uniform(20, 60), 1)
    }

@app.get("/api/external/weather")
async def get_weather_data(lat: float, lon: float):
    """Mocks NASA POWER API response for the location / season"""
    return {
        "temperature": round(random.uniform(20, 35), 2),
        "humidity": round(random.uniform(40, 80), 2),
        "rainfall_mm": round(random.uniform(500, 1500), 2),
        "solar_radiation": round(random.uniform(15, 25), 2)
    }

# --- Feature Processing / Synthesis ---

@app.post("/api/prepare")
async def prepare_features(data: GeolocationInput):
    """
    Acquires data from external APIs and synthesizes the final input vector
    for the recommendation model.
    """
    # In a real environment we would make async HTTP calls here.
    # We call the mock functions directly to bypass networking overhead for MVP.
    soil_data = await get_soil_data(data.lat, data.lon)
    weather_data = await get_weather_data(data.lat, data.lon)
        
    # Feature Engineering & Synthesis logic
    features = {
        "lat": data.lat,
        "lon": data.lon,
        **soil_data,
        **weather_data,
        "prev_crop": data.previous_crop,
        "prev_fert_n": data.previous_fertilizer_n,
        "prev_fert_p": data.previous_fertilizer_p,
        "prev_fert_k": data.previous_fertilizer_k
    }
    
    # Execute mass balance and data standardization
    features["adjusted_nitrogen"] = features["nitrogen"] + (features["prev_fert_n"] * 0.1)
    features["adjusted_phosphorus"] = features["prev_fert_p"] * 0.05
    features["adjusted_potassium"] = features["prev_fert_k"] * 0.08
    
    return {
        "status": "success",
        "features": features
    }
