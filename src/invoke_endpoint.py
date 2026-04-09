import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import requests
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

# --------------------------------------------------
# Endpoint details
# --------------------------------------------------
ENDPOINT_URL = "https://amazon-review-endpoint-60306249.qatarcentral.inference.ml.azure.com/score"
API_KEY = ""

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

# --------------------------------------------------
# Load dataset
# --------------------------------------------------
def load_dataset():
    ACCOUNT_NAME = "amazonlab260300152566028"
    ACCOUNT_KEY  = ""
    CONTAINER    = "azureml-blobstore-ce895e8b-6216-4417-abb9-bd9f29859019"
    BLOB_PATH    = "azureml/251bed10-b997-4fe7-b462-b7d17ae9e49c/output_data/data.parquet"

    abfs_path = f"abfs://{CONTAINER}/{BLOB_PATH}"
    print(f"Reading from: {abfs_path}")

    df = pd.read_parquet(
        abfs_path,
        storage_options={
            "account_name": ACCOUNT_NAME,
            "account_key": ACCOUNT_KEY
        }
    )
    print(f"Loaded shape: {df.shape}")
    return df


# --------------------------------------------------
# Create labels (must match training script)
# --------------------------------------------------
def create_labels(df):
    df["label"] = (df["overall"] >= 4).astype(int)
    return df


# --------------------------------------------------
# Build features (must match training script)
# --------------------------------------------------
def build_features(df):
    sbert_cols    = [c for c in df.columns if c.startswith('bert_embedding_')]
    tfidf_cols    = [c for c in df.columns if c.startswith('tfidf_')]
    sentiment_cols = ['sentiment_compound']
    length_cols   = ['review_length_words']

    feature_cols = sbert_cols + tfidf_cols + sentiment_cols + length_cols
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].fillna(0)
    print(f"Feature matrix shape: {X.shape}")
    return X


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    print("Loading deployment dataset...")
    df = load_dataset()
    df = create_labels(df)
    y_true = df["label"].values

    print("Building features...")
    X = build_features(df)

    batch_size = 100
    all_preds  = []

    print(f"Sending {len(X)} samples in batches of {batch_size}...")

    for i in range(0, len(X), batch_size):
        batch   = X.iloc[i:i + batch_size]
        payload = {"data": batch.to_dict(orient="records")}

        response = requests.post(ENDPOINT_URL, headers=headers, json=payload, verify=False)

        if response.status_code != 200:
            print(f"Batch {i} failed: {response.status_code} - {response.text}")
            continue

        all_preds.extend(response.json())

        if i % 1000 == 0:
            print(f"  Processed {i}/{len(X)} samples...")

    print(f"\nTotal predictions received: {len(all_preds)}")

    y_pred = np.array(all_preds)
    y_true = y_true[:len(y_pred)]

    print(f"\n===== DEPLOYMENT RESULTS =====")
    print(f"Accuracy  : {accuracy_score(y_true, y_pred):.4f}")
    print(f"F1        : {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Precision : {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall    : {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Samples   : {len(y_pred)}")


if __name__ == "__main__":
    main()