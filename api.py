from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("gw_model.pkl")
scaler = joblib.load("scaler.pkl")


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

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        input_df = pd.DataFrame([[data[col] for col in columns]], columns=columns)

        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]

        return jsonify({
            "prediction": float(prediction),
            "unit": "MBGL"
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)