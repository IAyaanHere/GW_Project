from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

# -----------------------------
# LOAD MODEL + SCALER
# -----------------------------
model = joblib.load("gw_model.pkl")
scaler = joblib.load("scaler.pkl")

# -----------------------------
# LOAD DATASET
# -----------------------------
data = pd.read_csv("Spatial_GW_Dataset_2015_Enhanced.csv")

# Ensure numeric
data = data.apply(pd.to_numeric, errors='coerce').dropna()

# -----------------------------
# FEATURE ORDER (VERY IMPORTANT)
# -----------------------------
columns = [
    "Latitude",
    "Longitude",
    "Rainfall",
    "GW_Recharge",
    "GW_Extraction",
    "Extraction_Stage_Perc",
    "Prev_GW"
]

# -----------------------------
# HOME
# -----------------------------
@app.route("/")
def home():
    return jsonify({"message": "Groundwater API Running 🚀"})


# -----------------------------
# 🚀 MAIN API (GPS → PREDICTION)
# -----------------------------
@app.route("/predict-from-location", methods=["POST"])
def predict_from_location():
    try:
        req = request.get_json()

        # ✅ Validate input
        if "Latitude" not in req or "Longitude" not in req:
            return jsonify({"error": "Latitude & Longitude required"}), 400

        lat = float(req["Latitude"])
        lon = float(req["Longitude"])

        # -----------------------------
        # FIND NEAREST LOCATION
        # -----------------------------
        df = data.copy()

        # Faster + no sqrt needed
        df["distance"] = (
            (df["Latitude"] - lat) ** 2 +
            (df["Longitude"] - lon) ** 2
        )

        nearest = df.loc[df["distance"].idxmin()]

        # -----------------------------
        # BUILD MODEL INPUT
        # -----------------------------
        input_data = [[
            lat,
            lon,
            float(nearest["Rainfall"]),
            float(nearest["GW_Recharge"]),
            float(nearest["GW_Extraction"]),
            float(nearest["Extraction_Stage_Perc"]),
            float(nearest["Prev_GW"])
        ]]

        input_df = pd.DataFrame(input_data, columns=columns)

        # -----------------------------
        # SCALE + PREDICT
        # -----------------------------
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]

        # -----------------------------
        # RESPONSE
        # -----------------------------
        return jsonify({
            "prediction": float(prediction),
            "unit": "MBGL",
            "nearest_data": {
                "Rainfall": float(nearest["Rainfall"]),
                "GW_Recharge": float(nearest["GW_Recharge"]),
                "GW_Extraction": float(nearest["GW_Extraction"]),
                "Extraction_Stage_Perc": float(nearest["Extraction_Stage_Perc"]),
                "Prev_GW": float(nearest["Prev_GW"])
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -----------------------------
# RUN (RENDER COMPATIBLE)
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
