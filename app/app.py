from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("model.pkl")

@app.route("/")
def home():
    return "Rail Delay Predictor API is running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    features = np.array([[
        data["hour"],
        data["day_of_week"],
        data["is_peak"],
        data["weather_delay_risk"]
    ]])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    return jsonify({
        "delayed_prediction": int(prediction),
        "delay_probability": float(probability)
    })

if __name__ == "__main__":
    app.run(debug=True)