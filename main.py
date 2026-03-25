import pickle
import numpy as np
import pandas as pd
import os

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load model and vectorizer
with open("Spam_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vectorizer_file:
    vectorize = pickle.load(vectorizer_file)


@app.route("/")
def home():
    return "Welcome! Spam Email Detection API is running."


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data
        data = request.get_json()

        if not data:
            return jsonify({"error": "No input data provided"}), 400

        required_columns = ["Body"]

        # Convert to DataFrame
        input_data = pd.DataFrame([data])

        # Check missing columns
        if not all(col in input_data.columns for col in required_columns):
            return jsonify({
                "error": f"Missing required columns. Required columns = {required_columns}"
            }), 400

        # Vectorize input
        vectorized_data = vectorize.transform(input_data["Body"])

        # Predict
        prediction = model.predict(vectorized_data)

        # Confidence calculation
        try:
            proba = model.predict_proba(vectorized_data)[0][1]
            confidence = float(proba * 100) if prediction[0] == 1 else float((1 - proba) * 100)
        except:
            confidence = 95.0

        # Response
        response = {
            "prediction": "Spam" if prediction[0] == 1 else "Ham",
            "is_spam": bool(prediction[0] == 1),
            "confidence": confidence
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Render-compatible run
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)