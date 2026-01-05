from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model at startup (BEST PRACTICE)
MODEL_PATH = "logistic_model.joblib"
model = joblib.load(MODEL_PATH)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "UP"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Expecting JSON input
        data = request.get_json()

        if "features" not in data:
            return jsonify({"error": "Missing 'features' field"}), 400

        # Convert input to numpy array
        features = np.array(data["features"]).reshape(1, -1)

        # Prediction
        prediction = model.predict(features)[0]

        # If probability is supported
        response = {"prediction": int(prediction)}
        if hasattr(model, "predict_proba"):
            response["probability"] = model.predict_proba(features).tolist()

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
