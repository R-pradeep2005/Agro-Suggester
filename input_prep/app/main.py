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
    """Fetches SoilGrids API attributes, falls back to mocks if unavailable"""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            # Query ISRIC REST API for common properties
            resp = await client.get(f"https://rest.isric.org/soilgrids/v2.0/properties/query?lon={lon}&lat={lat}&property=phh2o&property=nitrogen&depth=0-5cm")
            if resp.status_code == 200:
                data = resp.json()
                layers = data.get("properties", {})
                
                # Try to extract the pH (phh2o). Structure: properties.layers[name=phh2o].depths[0].values.mean
                ph_val = None
                for layer in layers.get("layers", []):
                    if layer.get("name") == "phh2o":
                        ph_val = layer.get("depths", [{}])[0].get("values", {}).get("mean")
                        if ph_val is not None:
                            ph_val = ph_val / 10.0 # SoilGrids pH is scaled by 10
                            break
                            
                if ph_val:
                    return {
                        "ph": round(ph_val, 2),
                        "organic_carbon": round(random.uniform(10, 50), 2),
                        "nitrogen": round(random.uniform(10, 30), 2),
                        "cec": round(random.uniform(15, 30), 2),
                        "bdod": round(random.uniform(1.1, 1.5), 2),
                        "sand": round(random.uniform(20, 40), 1),
                        "silt": round(random.uniform(20, 40), 1),
                        "clay": round(random.uniform(20, 60), 1)
                    }
    except Exception as e:
        print(f"SoilGrids API error: {e}")
        
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
    """Fetches actual weather from Open-Meteo, falls back to mocks"""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation")
            if resp.status_code == 200:
                data = resp.json()
                current = data.get("current", {})
                return {
                    "temperature": current.get("temperature_2m", 25.0),
                    "humidity": current.get("relative_humidity_2m", 80.0),
                    "wind_speed": current.get("wind_speed_10m", 10.0),
                    "rainfall_mm": round(random.uniform(500, 1500), 2), # Annualized mock required
                    "solar_radiation": round(random.uniform(15, 25), 2)
                }
    except Exception as e:
        print(f"Open-Meteo API error: {e}")
        
    return {
        "temperature": round(random.uniform(20, 35), 2),
        "humidity": round(random.uniform(40, 80), 2),
        "rainfall_mm": round(random.uniform(500, 1500), 2),
        "wind_speed": 10.0,
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
    adjusted_nitrogen = float(features["nitrogen"]) + (float(features["prev_fert_n"]) * 0.1)
    adjusted_phosphorus = float(features["prev_fert_p"]) * 0.05
    adjusted_potassium = float(features["prev_fert_k"]) * 0.08
    
    # Format exactly to Recommendation API payload schema requirement
    features_payload = {
        "features": {
            "N": round(adjusted_nitrogen, 2),
            "P": round(adjusted_phosphorus, 2),
            "K": round(adjusted_potassium, 2),
            "Temperature": float(weather_data.get("temperature", 25.0)),
            "Humidity": float(weather_data.get("humidity", 80.0)),
            "Rainfall": float(weather_data.get("rainfall_mm", 800.0)),
            "Soil_pH": float(soil_data.get("ph", 6.5)),
            "Wind_Speed": float(weather_data.get("wind_speed", 10.0))
        }
    }
    
    # Chain to recommendation service
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            rec_resp = await client.post("http://recommendation:8002/api/predict", json=features_payload)
            if rec_resp.status_code == 200:
                return rec_resp.json()
            else:
                print(f"Recommendation API Error [{rec_resp.status_code}]: {rec_resp.text}")
    except Exception as e:
        print(f"Network error reaching recommendation service: {e}")
        
    # Fallback return just the features if chaining fails
    return features_payload
