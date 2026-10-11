from flask import Flask, request, jsonify
import joblib, xgboost as xgb, pandas as pd, numpy as np, io

superkart_api = Flask(__name__)
preprocessor = joblib.load("preprocessor.joblib")
xgb_model = xgb.XGBRegressor()
xgb_model.load_model("xgb_model.json")
print("Model loaded successfully!")

@superkart_api.route("/", methods=["GET"])
def home():
    return jsonify({"message": "SuperKart Sales Prediction API", "version": "1.0"})

@superkart_api.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})

@superkart_api.post("/v1/predict")
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])
        transformed = preprocessor.transform(df)
        prediction = xgb_model.predict(transformed)
        return jsonify({"prediction": round(float(prediction[0]), 2), "status": "success"})
    except Exception as e:
        return jsonify({"error": str(e), "status": "failed"}), 400

@superkart_api.post("/v1/predictbatch")
def predict_batch():
    try:
        file = request.files["file"]
        input_df = pd.read_csv(io.StringIO(file.read().decode("utf-8")))
        transformed = preprocessor.transform(input_df)
        preds = xgb_model.predict(transformed)
        return jsonify({str(i): round(float(p), 2) for i, p in enumerate(preds)})
    except Exception as e:
        return jsonify({"error": str(e), "status": "failed"}), 400

if __name__ == "__main__":
    superkart_api.run(host="0.0.0.0", port=7860, debug=False)
