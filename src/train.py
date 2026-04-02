import argparse
import os
import time
import json
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
    parser.add_argument("--c", type=float, default=1.0, help="Inverse regularization strength")
    parser.add_argument("--max_iter", type=int, default=1000, help="Maximum iterations")
    return parser.parse_args()

# --------------------------------------------------
# Load data
# --------------------------------------------------
def load_data(path):
    """Load parquet from folder"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path does not exist: {path}")
    
    if os.path.isdir(path):
        parquet_path = os.path.join(path, "data.parquet")
    else:
        parquet_path = path
    
    return pd.read_parquet(parquet_path)

# --------------------------------------------------
# Labels
# --------------------------------------------------
def create_labels(df):
    """Convert overall rating to binary classification label"""
    if "overall" not in df.columns:
        raise RuntimeError("Column 'overall' is missing. You had one job.")
    
    # Binary classification: rating >= 4 is positive (1), else negative (0)
    # Overall is normalized 0-1, so >= 4 out of 5 becomes >= 0.8
    df["label"] = (df["overall"] >= 0.6).astype(int)
    return df

# --------------------------------------------------
# Features
# --------------------------------------------------
def build_features(df):
    """
    Construct feature matrix from merged features.
    SAME logic must be applied to train, val, and test.
    """
    
    # Use only strongest features: SBERT + TF-IDF + core sentiment + length
    sbert_cols = [col for col in df.columns if col.startswith('bert_embedding_')]
    tfidf_cols = [col for col in df.columns if col.startswith('tfidf_')]
    sentiment_cols = ['sentiment_compound']  # ← Only compound, drop pos/neg/neu (correlated)
    length_cols = ['review_length_words']    # ← Only words, drop chars (correlated)
    
    # Don't use derived sentiment columns (title_sentiment, brand_sentiment, etc.)
    # These are redundant - we already have sentiment on full reviewText
    
    # Collect feature columns
    feature_cols = sbert_cols + tfidf_cols + sentiment_cols + length_cols
    
    # Verify all exist
    missing_cols = [col for col in sentiment_cols + length_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns {missing_cols}, proceeding with available features")
        feature_cols = [col for col in feature_cols if col in df.columns]
    
    # Extract feature matrix
    X = df[feature_cols].copy()
    
    # Handle missing values
    X = X.fillna(0)
    
    # Check for empty matrix
    if len(X) == 0:
        raise RuntimeError("Feature matrix is empty. Impressive.")
    
    if X.shape[1] == 0:
        raise RuntimeError("No features selected. Check your feature columns.")
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Feature columns: {X.shape[1]}")
    
    return X, feature_cols

# --------------------------------------------------
# Evaluation
# --------------------------------------------------
def evaluate(model, X, y, split, scaler=None):
    """Evaluate model and log metrics"""
    
    # Apply scaler if provided (for val/test, use train's scaler)
    if scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X
    
    # Generate predictions
    preds = model.predict(X_scaled)
    proba = model.predict_proba(X_scaled)[:, 1]  # Probability of positive class
    
    # Calculate metrics
    acc = accuracy_score(y, preds)
    auc = roc_auc_score(y, proba)
    precision = precision_score(y, preds, zero_division=0)
    recall = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)
    
    # Log metrics
    mlflow.log_metric(f"{split}_accuracy", acc)
    mlflow.log_metric(f"{split}_auc", auc)
    mlflow.log_metric(f"{split}_precision", precision)
    mlflow.log_metric(f"{split}_recall", recall)
    mlflow.log_metric(f"{split}_f1", f1)
    
    print(f"{split.upper()} METRICS:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  AUC:       {auc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    
    return {
        'accuracy': acc,
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    args = parse_args()
    
    # Start timer for runtime logging
    start_time = time.time()
    
    print("=" * 80)
    print("TRAINING SCRIPT")
    print("=" * 80)
    
    # Load datasets
    print("\nLoading data...")
    train_df = load_data(args.train_data)
    val_df = load_data(args.val_data)
    test_df = load_data(args.test_data)
    
    print(f"Train shape: {train_df.shape}")
    print(f"Val shape: {val_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    # Create labels
    print("\nCreating labels...")
    train_df = create_labels(train_df)
    val_df = create_labels(val_df)
    test_df = create_labels(test_df)
    
    print(f"Train label distribution:\n{train_df['label'].value_counts()}")
    print(f"Val label distribution:\n{val_df['label'].value_counts()}")
    print(f"Test label distribution:\n{test_df['label'].value_counts()}")

    # Check label distribution
    print(f"\nTrain label distribution:\n{train_df['label'].value_counts()}")
    print(f"\nLabel value counts (details):")
    print(train_df['label'].value_counts(dropna=False))

    # STOP if only one class
    if len(train_df['label'].unique()) == 1:
        raise RuntimeError(f"ERROR: Only one class in training data! Classes: {train_df['label'].unique()}")
    
    # Build features using EXACT SAME logic for all splits
    print("\nBuilding features...")
    X_train, feature_cols = build_features(train_df)
    y_train = train_df["label"]
    
    X_val, _ = build_features(val_df)
    y_val = val_df["label"]
    
    X_test, _ = build_features(test_df)
    y_test = test_df["label"]
    
    # Verify all splits have same features
    assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1], \
        "Feature dimensions don't match across splits!"
    
    # Standardize features
    print("\nStandardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Log hyperparameters
    print("\nLogging hyperparameters...")
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("C", args.c)
    mlflow.log_param("max_iter", args.max_iter)
    mlflow.log_param("num_features", X_train.shape[1])
    mlflow.log_param("train_samples", X_train.shape[0])
    mlflow.log_param("val_samples", X_val.shape[0])
    mlflow.log_param("test_samples", X_test.shape[0])
    
    # Train model
    print("\nTraining model...")
    model = LogisticRegression(
        C=0.02,  # ← Stronger regularization
        class_weight='balanced',  # ← Handle imbalance
        max_iter=args.max_iter,
        random_state=42,
        verbose=0
    )
    model.fit(X_train_scaled, y_train)
    print("Model training complete!")
    
    # Evaluate on all splits
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    
    train_metrics = evaluate(model, X_train_scaled, y_train, "train")
    val_metrics = evaluate(model, X_val_scaled, y_val, "val")
    test_metrics = evaluate(model, X_test_scaled, y_test, "test")
    
    # Save model
    print("\nSaving model...")
    os.makedirs(args.output, exist_ok=True)
    model_path = os.path.join(args.output, "model.pkl")
    
    # Save both model and scaler for inference
    model_artifacts = {
        'model': model,
        'scaler': scaler,
        'feature_cols': feature_cols
    }
    joblib.dump(model_artifacts, model_path)
    # mlflow.log_artifact(model_path)
    print(f"Model saved to: {model_path}")
    
    # Log runtime
    runtime_seconds = time.time() - start_time
    mlflow.log_metric("training_runtime_seconds", runtime_seconds)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Total runtime: {runtime_seconds:.2f} seconds ({runtime_seconds/60:.2f} minutes)")
    print(f"Model saved to: {model_path}")

if __name__ == "__main__":
    main()