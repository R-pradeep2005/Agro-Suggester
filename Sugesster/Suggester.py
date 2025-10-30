import joblib
import pandas as pd
import shap
import numpy as np

# Crop model configurations
CROPS = {
    "rice": {"model": "models/rice.pkl", "data": "data-sets/rice_yield_dataset.csv"},
    "maize": {"model": "models/maize.pkl", "data": "data-sets/maize_yield_dataset.csv"},
    "wheat": {"model": "models/wheat.pkl", "data": "data-sets/wheat_yield_dataset.csv"},
    "groundnut": {"model": "models/groundnut.pkl", "data": "data-sets/groundnut_yield_dataset.csv"},
    "sugarcane": {"model": "models/cotton.pkl", "data": "data-sets/cotton_yield_dataset.csv"}
}

# ---- SAMPLE INPUT ----
input_data = {
  "temp": 28.52, "humidity": 46.8, "rainfall": 832.6, "solar_radiation": 16.61,
  "soil_bdod": 144.5, "soil_cec": 19.49, "soil_cfvo": 10.61, "soil_clay": 173.4,
  "soil_sand": 377.0, "soil_silt": 276.8, "soil_nitrogen": 158.8, "soil_ocd": 44.1,
  "soil_ocs": 36.6, "soil_phh2o": 8.45, "soil_soc": 24.73, "soil_wv0010": 33.78,
  "previous_crop": "maize", "previous_fertilizer": "NPK",
  "fertilizer_amount": 249.7, "cultivation_break": 0
}
input_df_raw = pd.DataFrame([input_data])

# ---- Function for individual crop prediction & explainability ----
def predict_and_explain(crop_name, cfg):
    try:
        model = joblib.load(cfg["model"])
        background_data = pd.read_csv(cfg["data"])
    except FileNotFoundError:
        print(f"❌ Missing files for {crop_name}, skipping...")
        return None

    # One-hot encode both input and dataset
    df_bg = pd.get_dummies(background_data, columns=['previous_crop', 'previous_fertilizer'])
    df_input = pd.get_dummies(input_df_raw, columns=['previous_crop', 'previous_fertilizer'])

    if hasattr(model, "get_booster"):
    # XGBoost model
        expected = model.get_booster().feature_names
    else:
    # scikit-learn model (like RandomForest, DecisionTree)
        expected = model.feature_names_in_ if hasattr(model, "feature_names_in_") else None

    df_bg = df_bg.reindex(columns=expected, fill_value=0)
    df_input = df_input.reindex(columns=expected, fill_value=0)

    # Predict yield
    pred_yield = float(model.predict(df_input)[0])

    # SHAP explanation
    sample = df_bg.sample(n=min(100, len(df_bg)), random_state=42)
    explainer = shap.Explainer(model.predict, sample)
    shap_vals = explainer(df_input)
    shap_matrix = shap_vals.values
    shap_abs = np.abs(shap_matrix).flatten()

    # Extract top 3 positive features
    factors = sorted(zip(df_input.columns, shap_matrix.flatten()), key=lambda x: abs(x[1]), reverse=True)[:5]
    pos_factors = [f for f, s in factors if s > 0][:3]

    # Mitigation strategy summary
    strategies = []
    for f in pos_factors:
        fname = f.lower()
        if "fertilizer" in fname or "nitrogen" in fname:
            strategies.append(f"{f}: Maintain balanced fertilizer schedule and optimize nutrient timing.")
        elif "soil" in fname:
            strategies.append(f"{f}: Continue improving soil health with organic matter and pH balance.")
        elif "rain" in fname or "humid" in fname:
            strategies.append(f"{f}: Ensure irrigation matches crop water needs.")
        elif "temp" in fname or "solar" in fname:
            strategies.append(f"{f}: Choose heat-tolerant variety or adjust planting schedule.")
        else:
            strategies.append(f"{f}: Monitor this factor and adjust as advised by agronomist.")

    return {
        "crop": crop_name.capitalize(),
        "yield": pred_yield,
        "positive_factors": pos_factors,
        "strategies": strategies
    }

# ---- Run across all crops ----
results = []
for crop, cfg in CROPS.items():
    res = predict_and_explain(crop, cfg)
    if res: results.append(res)

# ---- Sort and display top 3 ----
top3 = sorted(results, key=lambda x: x["yield"], reverse=True)[:3]

print("\n🌾 Top 3 Crop Recommendations:")
for i, r in enumerate(top3, 1):
    print(f"\n{i}. {r['crop']} — Predicted Yield: {r['yield']:.2f} tons/ha")
    print(f"   Positive Factors: {', '.join(r['positive_factors']) if r['positive_factors'] else 'N/A'}")
    print("   Mitigation Strategies:")
    for s in r['strategies']:
        print(f"    - {s}")
