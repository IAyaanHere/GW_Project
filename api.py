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

# Feature order (IMPORTANT)
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
# AUTO-FILL (Nearest Location)
# -----------------------------
@app.route("/auto-fill", methods=["POST"])
def auto_fill():
    try:
        req = request.get_json()

        lat = float(req["Latitude"])
        lon = float(req["Longitude"])

        # Copy dataset (important)
        df = data.copy()

        # Distance calculation
        df["distance"] = np.sqrt(
            (df["Latitude"] - lat) ** 2 +
            (df["Longitude"] - lon) ** 2
        )

        nearest = df.loc[df["distance"].idxmin()]

        return jsonify({
            "Rainfall": float(nearest["Rainfall"]),
            "Prev_GW": float(nearest["Prev_GW"]),
            "GW_Recharge": float(nearest["GW_Recharge"]),
            "GW_Extraction": float(nearest["GW_Extraction"]),
            "Extraction_Stage_Perc": float(nearest["Extraction_Stage_Perc"])
        })

    except Exception as e:
        return jsonify({"error": str(e)})


# -----------------------------
# PREDICT
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data_req = request.get_json()

        input_data = [
            float(data_req[col]) for col in columns
        ]

        input_df = pd.DataFrame([input_data], columns=columns)

        input_scaled = scaler.transform(input_df)

        prediction = model.predict(input_scaled)[0]

        return jsonify({
            "prediction": float(prediction),
            "unit": "MBGL"
        })

    except Exception as e:
        return jsonify({"error": str(e)})


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)