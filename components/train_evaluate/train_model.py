import argparse
import os
import time
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train RandomForest for RUL prediction and evaluate")
    parser.add_argument("--input_train",       type=str, required=True,
                        help="Parquet folder — training split (features + target_RUL)")
    parser.add_argument("--input_val",         type=str, required=True,
                        help="Parquet folder — validation split (features + target_RUL)")
    parser.add_argument("--output_model",      type=str, required=True,
                        help="Output folder: saved model (.joblib)")
    parser.add_argument("--output_metrics",    type=str, required=True,
                        help="Output folder: metrics JSON and prediction CSV")
    # Model hyperparameters
    parser.add_argument("--n_estimators",      type=int,   default=200)
    parser.add_argument("--max_depth",         type=int,   default=None,
                        help="Max tree depth (None = unlimited)")
    parser.add_argument("--min_samples_split", type=int,   default=5)
    parser.add_argument("--min_samples_leaf",  type=int,   default=2)
    parser.add_argument("--max_features",      type=str,   default="sqrt",
                        help="Max features per split: 'sqrt', 'log2', or float 0-1")
    parser.add_argument("--n_jobs",            type=int,   default=-1)
    parser.add_argument("--random_seed",       type=int,   default=42)
    return parser.parse_args()


def load_parquet_folder(folder_path: str) -> pd.DataFrame:
    files = [f for f in os.listdir(folder_path) if f.endswith(".parquet")]
    if not files:
        raise FileNotFoundError(f"No parquet files in {folder_path}")
    return pd.concat(
        [pd.read_parquet(os.path.join(folder_path, f)) for f in files],
        ignore_index=True
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute RMSE, MAE, R², and score distribution."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    errors = y_pred - y_true

    # Asymmetric scoring function (penalises late predictions more)
    def nasa_score(errors):
        scores = np.where(
            errors < 0,
            np.exp(-errors / 13) - 1,
            np.exp(errors / 10) - 1
        )
        return float(np.sum(scores))

    return {
        "RMSE": float(rmse),
        "MAE":  float(mae),
        "R2":   float(r2),
        "NASA_score": nasa_score(errors),
        "mean_error":  float(errors.mean()),
        "std_error":   float(errors.std()),
        "n_samples":   int(len(y_true)),
    }


def main():
    args = parse_args()
    start = time.time()

    print("=" * 60)
    print("COMPONENT: train_evaluate")
    print("=" * 60)

    # --- Load ---
    print("\n[1/4] Loading data ...")
    train_df = load_parquet_folder(args.input_train)
    val_df   = load_parquet_folder(args.input_val)

    label_col = "target_RUL"
    meta_cols  = ["entity_id", label_col]
    feature_cols = [c for c in train_df.columns if c not in meta_cols]

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df[label_col].values.astype(np.float32)
    X_val   = val_df[feature_cols].values.astype(np.float32)
    y_val   = val_df[label_col].values.astype(np.float32)

    # Sanitise
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_val   = np.nan_to_num(X_val,   nan=0.0, posinf=0.0, neginf=0.0)

    print(f"  X_train: {X_train.shape}, y_train range: "
          f"[{y_train.min():.0f}, {y_train.max():.0f}]")
    print(f"  X_val:   {X_val.shape},   y_val range: "
          f"[{y_val.min():.0f}, {y_val.max():.0f}]")
    print(f"  Features: {len(feature_cols)}")

    # Handle max_depth arg (None or int)
    max_depth = args.max_depth if args.max_depth and args.max_depth > 0 else None

    # Handle max_features arg (string or float)
    max_features = args.max_features
    try:
        mf_float = float(max_features)
        if 0.0 < mf_float <= 1.0:
            max_features = mf_float
    except ValueError:
        pass  # keep as string "sqrt" / "log2"

    # --- Train ---
    print(f"\n[2/4] Training RandomForestRegressor "
          f"(n_estimators={args.n_estimators}, max_depth={max_depth}) ...")
    t0 = time.time()
    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        max_features=max_features,
        n_jobs=args.n_jobs,
        random_state=args.random_seed,
        oob_score=True,
    )
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"  Training done in {train_time:.1f}s")
    print(f"  OOB R²: {model.oob_score_:.4f}")

    # --- Evaluate ---
    print("\n[3/4] Evaluating ...")
    y_train_pred = model.predict(X_train)
    y_val_pred   = model.predict(X_val)

    train_metrics = compute_metrics(y_train, y_train_pred)
    val_metrics   = compute_metrics(y_val,   y_val_pred)

    print(f"\n  TRAIN — RMSE: {train_metrics['RMSE']:.2f}  "
          f"MAE: {train_metrics['MAE']:.2f}  R²: {train_metrics['R2']:.4f}")
    print(f"  VAL   — RMSE: {val_metrics['RMSE']:.2f}  "
          f"MAE: {val_metrics['MAE']:.2f}  R²: {val_metrics['R2']:.4f}  "
          f"NASA_score: {val_metrics['NASA_score']:.2f}")

    # Feature importances (top 20)
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    top_features = importances.sort_values(ascending=False).head(20).to_dict()

    # --- Save ---
    print("\n[4/4] Saving model and metrics ...")
    os.makedirs(args.output_model,   exist_ok=True)
    os.makedirs(args.output_metrics, exist_ok=True)

    # Model
    model_path = os.path.join(args.output_model, "random_forest_rul.joblib")
    joblib.dump(model, model_path)
    print(f"  Model saved → {model_path}")

    # Metrics JSON
    metrics_payload = {
        "train": train_metrics,
        "val":   val_metrics,
        "training_time_seconds": train_time,
        "n_features": len(feature_cols),
        "oob_r2": float(model.oob_score_),
        "hyperparameters": {
            "n_estimators":      args.n_estimators,
            "max_depth":         str(max_depth),
            "min_samples_split": args.min_samples_split,
            "min_samples_leaf":  args.min_samples_leaf,
            "max_features":      str(max_features),
            "random_seed":       args.random_seed,
        },
        "top_20_features_by_importance": top_features,
    }
    metrics_path = os.path.join(args.output_metrics, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)

    # Predictions CSV
    preds_df = pd.DataFrame({
        "entity_id":  val_df["entity_id"].values,
        "y_true":     y_val,
        "y_pred":     y_val_pred,
        "error":      y_val_pred - y_val,
    })
    preds_df.to_csv(os.path.join(args.output_metrics, "val_predictions.csv"), index=False)

    # Feature importance CSV
    importances.sort_values(ascending=False).reset_index().rename(
        columns={"index": "feature", 0: "importance"}
    ).to_csv(os.path.join(args.output_metrics, "feature_importances.csv"), index=False)

    elapsed = time.time() - start
    print(f"\n✅ train_evaluate complete in {elapsed:.1f}s")
    print(f"   Val RMSE:       {val_metrics['RMSE']:.4f}")
    print(f"   Val R²:         {val_metrics['R2']:.4f}")
    print(f"   Val NASA score: {val_metrics['NASA_score']:.2f}")
    print(f"   Metrics → {metrics_path}")


if __name__ == "__main__":
    main()