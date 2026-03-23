import random
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List

app = FastAPI(title="Agro-Suggester Recommendation Service")

class PredictionRequest(BaseModel):
    features: Dict[str, Any]

@app.get("/health")
async def health():
    return {"status": "ok", "service": "recommendation"}

# --- Mock Model Registry & Loading ---
AVAILABLE_CROPS = ["Rice", "Wheat", "Maize", "Cotton", "Sugarcane", "Tomato", "Soybean"]

def load_crop_model(crop_name: str):
    """Mocks loading a serialized .pkl model for a specific crop"""
    return {"model_id": f"xgb_{crop_name.lower()}", "status": "loaded"}

# --- Prediction and Explainability ---
def predict_yield(crop_model, features: Dict[str, Any]) -> float:
    """Mocks yield prediction based on features"""
    # Base yield plus some random variance heavily influenced by nitrogen and rainfall
    base_yield = random.uniform(2000, 5000)
    variance = (features.get("adjusted_nitrogen", 0) * 10) + (features.get("rainfall_mm", 0) * 0.5)
    return round(base_yield + variance, 2)

def generate_shap_explanation(crop_name: str, features: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Mocks SHAP feature attribution"""
    explanations = [
        {"feature": "adjusted_nitrogen", "impact": round(random.uniform(0.1, 0.5), 2), "direction": "positive"},
        {"feature": "rainfall_mm", "impact": round(random.uniform(-0.3, 0.4), 2), "direction": "positive" if random.choice([True, False]) else "negative"},
        {"feature": "temperature", "impact": round(random.uniform(-0.2, 0.2), 2), "direction": "negative"},
    ]
    # Sort by absolute impact
    return sorted(explanations, key=lambda x: abs(x["impact"]), reverse=True)

@app.post("/api/predict")
async def predict_recommendation(data: PredictionRequest):
    """
    Evaluates all crop models and returns top 3 recommendations with XAI insights.
    """
    predictions = []
    
    # 1. Evaluate independent crop models
    for crop in AVAILABLE_CROPS:
        model = load_crop_model(crop)
        predicted_yield = predict_yield(model, data.features)
        shap_explanation = generate_shap_explanation(crop, data.features)
        
        # Mapping SHAP to human rules
        insight_string = f"Optimal nitrogen levels contribute positively."
        if shap_explanation[0]["feature"] == "rainfall_mm" and shap_explanation[0]["direction"] == "negative":
            insight_string = "High vulnerability to projected rainfall deviations."
            
        predictions.append({
            "crop": crop,
            "predicted_yield_kg_ha": predicted_yield,
            "explanation": shap_explanation,
            "insight": insight_string
        })
        
    # 2. Rank crops by yield
    predictions.sort(key=lambda x: x["predicted_yield_kg_ha"], reverse=True)
    
    # 3. Return top 3
    top_3 = predictions[:3]
    
    return {
        "status": "success",
        "recommendations": top_3
    }
