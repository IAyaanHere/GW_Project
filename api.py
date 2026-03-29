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

# Load dataset for auto-fill
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

# Home route
@app.route("/")
def home():
    return "Groundwater Prediction API is running"

# -----------------------------
#  AUTO-FILL 
# -----------------------------
@app.route("/auto-fill", methods=["POST"])
def auto_fill():
    try:
        req = request.get_json()
        lat = req["Latitude"]
        lon = req["Longitude"]

        # Calculate distance
        data["distance"] = np.sqrt(
            (data["Latitude"] - lat)**2 +
            (data["Longitude"] - lon)**2
        )

        nearest = data.loc[data["distance"].idxmin()]

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
# 🔥 PREDICT ROUTE
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data_req = request.get_json()

        input_df = pd.DataFrame(
            [[data_req[col] for col in columns]],
            columns=columns
        )

        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]

        return jsonify({
            "prediction": float(prediction),
            "unit": "MBGL"
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# Run app
if __name__ == "__main__":
    app.run(debug=True)