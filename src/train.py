import argparse
import os
import time
import azureml.mlflow
import mlflow
import joblib
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# Arguments
# --------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--c", type=float, default=0.02)
    parser.add_argument("--max_iter", type=int, default=2000)
    return parser.parse_args()


# --------------------------------------------------
# Load data (FIXED)
# --------------------------------------------------
def load_data(path):
    print(f"Loading from path: {path}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Path does not exist: {path}")

    # Try both possibilities (Azure sometimes behaves weirdly)
    parquet_path = os.path.join(path, "data.parquet")

    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
    elif path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        raise FileNotFoundError(f"Could not find parquet file in {path}")

    print(f"Loaded shape: {df.shape}")
    return df


# --------------------------------------------------
# Labels
# --------------------------------------------------
def create_labels(df):
    if "overall" not in df.columns:
        raise RuntimeError("Column 'overall' is missing.")

    df["label"] = (df["overall"] >= 0.6).astype(int)
    return df


# --------------------------------------------------
# Features
# --------------------------------------------------
def build_features(df):
    sbert_cols = [c for c in df.columns if c.startswith('bert_embedding_')]
    tfidf_cols = [c for c in df.columns if c.startswith('tfidf_')]

    sentiment_cols = ['sentiment_compound']
    length_cols = ['review_length_words']

    feature_cols = sbert_cols + tfidf_cols + sentiment_cols + length_cols

    # Remove missing columns safely
    feature_cols = [c for c in feature_cols if c in df.columns]

    if len(feature_cols) == 0:
        raise RuntimeError("No valid features found.")

    X = df[feature_cols].fillna(0)

    print(f"Feature matrix shape: {X.shape}")

    return X, feature_cols


# --------------------------------------------------
# Evaluation
# --------------------------------------------------
def evaluate(model, X, y, split):
    preds = model.predict(X)
    proba = model.predict_proba(X)[:, 1]

    acc = accuracy_score(y, preds)
    auc = roc_auc_score(y, proba)
    precision = precision_score(y, preds, zero_division=0)
    recall = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)

    mlflow.log_metric(f"{split}_accuracy", acc)
    mlflow.log_metric(f"{split}_auc", auc)
    mlflow.log_metric(f"{split}_precision", precision)
    mlflow.log_metric(f"{split}_recall", recall)
    mlflow.log_metric(f"{split}_f1", f1)

    print(f"{split.upper()} → Acc: {acc:.4f}, F1: {f1:.4f}")

    return acc


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    args = parse_args()
    start_time = time.time()

    print("\n===== LOADING DATA =====")
    train_df = load_data(args.train_data)
    val_df = load_data(args.val_data)
    test_df = load_data(args.test_data)

    print("\n===== LABELS =====")
    train_df = create_labels(train_df)
    val_df = create_labels(val_df)
    test_df = create_labels(test_df)

    if len(train_df['label'].unique()) == 1:
        raise RuntimeError("Only one class in training data!")

    print("\n===== FEATURES =====")
    X_train, feature_cols = build_features(train_df)
    X_val, _ = build_features(val_df)
    X_test, _ = build_features(test_df)

    y_train = train_df["label"]
    y_val = val_df["label"]
    y_test = test_df["label"]

    print("\n===== SCALING =====")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print("\n===== LOGGING PARAMS =====")
    mlflow.log_param("C", args.c)
    mlflow.log_param("max_iter", args.max_iter)
    mlflow.log_param("num_features", X_train.shape[1])

    print("\n===== TRAINING =====")
    model = LogisticRegression(
        C=args.c,
        class_weight='balanced',
        max_iter=args.max_iter,
        random_state=123
    )
    model.fit(X_train, y_train)

    print("\n===== EVALUATION =====")
    evaluate(model, X_train, y_train, "train")
    evaluate(model, X_val, y_val, "val")
    evaluate(model, X_test, y_test, "test")

    print("\n===== SAVING MODEL =====")
    os.makedirs(args.output, exist_ok=True)

    model_path = os.path.join(args.output, "model.pkl")

    joblib.dump({
        "model": model,
        "scaler": scaler,
        "feature_cols": feature_cols
    }, model_path)

    print(f"Model saved to: {model_path}")

    runtime = time.time() - start_time
    mlflow.log_metric("training_runtime_seconds", runtime)

    print(f"\nDONE in {runtime:.2f} sec")


if __name__ == "__main__":
    main()