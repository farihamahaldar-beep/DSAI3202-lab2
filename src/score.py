import json
import numpy as np
import pandas as pd
import joblib
import os

# Global variables
model = None
scaler = None
feature_cols = None


# --------------------------------------------------
# Init (runs once when container starts)
# --------------------------------------------------
def init():
    global model, scaler, feature_cols

    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model.pkl")

    artifacts = joblib.load(model_path)

    model = artifacts["model"]
    scaler = artifacts["scaler"]
    feature_cols = artifacts["feature_cols"]

    print("Model loaded successfully")


# --------------------------------------------------
# Run (called per request)
# --------------------------------------------------
def run(raw_data):
    try:
        data = json.loads(raw_data)

        # Convert to DataFrame
        df = pd.DataFrame(data["data"])

        # Ensure all expected features exist
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0

        # Reorder columns to match training
        df = df[feature_cols]

        # Fill missing values
        df = df.fillna(0)

        # Scale
        X = scaler.transform(df)

        # Predict
        preds = model.predict(X)

        return preds.tolist()

    except Exception as e:
        return {"error": str(e)}