import argparse
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import joblib

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_features", type=str, required=True)
    parser.add_argument("--train_target", type=str, required=True)
    parser.add_argument("--val_features", type=str, required=True)
    parser.add_argument("--val_target", type=str, required=True)
    parser.add_argument("--test_features", type=str, required=True)
    parser.add_argument("--test_target", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--random_state", type=int, default=42)
    return parser.parse_args()

def load_data(path, filename):
    """Load parquet or CSV file"""
    if os.path.isdir(path):
        parquet_path = os.path.join(path, filename)
        if os.path.exists(parquet_path):
            if filename.endswith('.parquet'):
                return pd.read_parquet(parquet_path)
            else:
                return pd.read_csv(parquet_path)
    raise FileNotFoundError(f"Could not find {filename} in {path}")

def main():
    args = parse_args()
    
    print("Loading train/val/test data...")
    X_train = load_data(args.train_features, "train_features.parquet")
    y_train = load_data(args.train_target, "train_target.csv")['RUL'].values
    
    X_val = load_data(args.val_features, "val_features.parquet")
    y_val = load_data(args.val_target, "val_target.csv")['RUL'].values
    
    X_test = load_data(args.test_features, "test_features.parquet")
    y_test = load_data(args.test_target, "test_target.csv")['RUL'].values
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Train Random Forest model
    print(f"Training RandomForest with {args.n_estimators} estimators...")
    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Predict on train, val, and test
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"\nTrain RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f}")
    print(f"Val RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(args.output_dir, "model.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Save predictions
    pd.DataFrame({'actual': y_test, 'predicted': y_test_pred}).to_csv(
        os.path.join(args.output_dir, "test_predictions.csv"), index=False
    )
    
    # Save metrics
    metrics = {
        "train": {
            "rmse": float(train_rmse),
            "mae": float(train_mae),
            "r2": float(train_r2)
        },
        "validation": {
            "rmse": float(val_rmse),
            "mae": float(val_mae),
            "r2": float(val_r2)
        },
        "test": {
            "rmse": float(test_rmse),
            "mae": float(test_mae),
            "r2": float(test_r2)
        },
        "model_params": {
            "n_estimators": args.n_estimators,
            "n_features": X_train.shape[1],
            "random_state": args.random_state
        }
    }
    
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✅ Training complete!")

if __name__ == "__main__":
    main()