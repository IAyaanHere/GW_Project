from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

# Load model & scaler
model = joblib.load("gw_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load dataset
data = pd.read_csv("Spatial_GW_Dataset_2015_Enhanced.csv")

# Feature order
columns = [
    "Latitude",
    "Longitude",
    "Rainfall",
    "GW_Recharge",
    "GW_Extraction",
    "Extraction_Stage_Perc",
    "Prev_GW"
]

@app.route("/")
def home():
    return "Groundwater Prediction API is running"

# 🚀 ALL-IN-ONE ROUTE
@app.route("/predict-from-location", methods=["POST"])
def predict_from_location():
    try:
        req = request.get_json()

        lat = float(req.get("Latitude"))
        lon = float(req.get("Longitude"))

        if lat is None or lon is None:
            return jsonify({"error": "Latitude and Longitude required"}), 400

        # 🔥 Find nearest data
        df = data.copy()

        df["distance"] = (
            (df["Latitude"] - lat)**2 +
            (df["Longitude"] - lon)**2
        )

        nearest = df.loc[df["distance"].idxmin()]

        # 🔥 Prepare input
        input_values = [[
            lat,
            lon,
            nearest["Rainfall"],
            nearest["GW_Recharge"],
            nearest["GW_Extraction"],
            nearest["Extraction_Stage_Perc"],
            nearest["Prev_GW"]
        ]]

        input_df = pd.DataFrame(input_values, columns=columns)

        # 🔥 Scale + Predict
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]

        return jsonify({
            "prediction": float(prediction),
            "unit": "MBGL",

            # 👇 extra info (frontend ke liye)
            "auto_filled_data": {
                "Rainfall": float(nearest["Rainfall"]),
                "Prev_GW": float(nearest["Prev_GW"]),
                "GW_Recharge": float(nearest["GW_Recharge"]),
                "GW_Extraction": float(nearest["GW_Extraction"]),
                "Extraction_Stage_Perc": float(nearest["Extraction_Stage_Perc"])
            },

            "nearest_location": {
                "Latitude": float(nearest["Latitude"]),
                "Longitude": float(nearest["Longitude"])
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)