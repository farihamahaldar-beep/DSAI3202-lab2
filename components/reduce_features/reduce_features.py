import argparse
import os
import time
import json
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression


def parse_args():
    parser = argparse.ArgumentParser(description="Filter-based feature reduction: variance, correlation, MI")
    parser.add_argument("--input_train_features", type=str, required=True)
    parser.add_argument("--input_test_features",  type=str, required=True)
    parser.add_argument("--input_labels",          type=str, required=True)
    parser.add_argument("--output_train_features", type=str, required=True)
    parser.add_argument("--output_test_features",  type=str, required=True)
    parser.add_argument("--output_selected_cols",  type=str, required=True,
                        help="Output folder: JSON list of selected column names")
    parser.add_argument("--variance_threshold",    type=float, default=0.01,
                        help="Minimum variance to keep a feature")
    parser.add_argument("--correlation_threshold", type=float, default=0.95,
                        help="Drop one of two features if |corr| >= this value")
    parser.add_argument("--mi_top_k",              type=int,   default=100,
                        help="Number of top mutual-information features to retain")
    return parser.parse_args()


def load_parquet_folder(folder_path: str) -> pd.DataFrame:
    files = [f for f in os.listdir(folder_path) if f.endswith(".parquet")]
    if not files:
        raise FileNotFoundError(f"No parquet files found in {folder_path}")
    return pd.concat(
        [pd.read_parquet(os.path.join(folder_path, f)) for f in files],
        ignore_index=True
    )


def variance_filter(X: pd.DataFrame, threshold: float) -> list:
    """Remove near-zero variance features."""
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X)
    kept = X.columns[selector.get_support()].tolist()
    print(f"  Variance filter: {len(X.columns)} → {len(kept)} features "
          f"(removed {len(X.columns) - len(kept)})")
    return kept


def correlation_filter(X: pd.DataFrame, threshold: float) -> list:
    """Remove highly correlated features (keep first in each correlated group)."""
    corr_matrix = X.corr().abs()
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop = [col for col in upper_tri.columns if any(upper_tri[col] >= threshold)]
    kept = [c for c in X.columns if c not in to_drop]
    print(f"  Correlation filter (threshold={threshold}): "
          f"{len(X.columns)} → {len(kept)} features "
          f"(removed {len(to_drop)})")
    return kept


def mutual_info_filter(X: pd.DataFrame, y: pd.Series, top_k: int) -> list:
    """Keep top-k features by mutual information with the target."""
    top_k = min(top_k, len(X.columns))
    mi_scores = mutual_info_regression(X, y, n_neighbors=5, random_state=42)
    mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
    kept = mi_series.head(top_k).index.tolist()
    print(f"  Mutual-info filter (top_k={top_k}): "
          f"{len(X.columns)} → {len(kept)} features")
    print(f"  Top-5 MI features: {kept[:5]}")
    return kept


def main():
    args = parse_args()
    start = time.time()

    print("=" * 60)
    print("COMPONENT: reduce_features")
    print("=" * 60)

    # --- Load ---
    print("\n[1/4] Loading data ...")
    train_df  = load_parquet_folder(args.input_train_features)
    test_df   = load_parquet_folder(args.input_test_features)
    labels_df = load_parquet_folder(args.input_labels)

    # Separate metadata columns from feature columns
    meta_cols = ["entity_id"]
    feature_cols = [c for c in train_df.columns if c not in meta_cols]

    X_train = train_df[feature_cols].copy()
    X_test  = test_df[[c for c in feature_cols if c in test_df.columns]].copy()
    y_train = labels_df.set_index("entity_id")["target_RUL"].reindex(
        train_df["entity_id"]
    ).values

    print(f"  Input feature columns: {len(feature_cols)}")

    # Replace inf with NaN then fill with column median
    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test.replace([np.inf,  -np.inf], np.nan, inplace=True)
    X_train.fillna(X_train.median(), inplace=True)
    X_test.fillna(X_train.median(), inplace=True)   # use train medians for test

    # --- Step 1: Variance filter ---
    print("\n[2/4] Variance filtering ...")
    kept_after_var = variance_filter(X_train, args.variance_threshold)
    X_train = X_train[kept_after_var]
    X_test  = X_test[[c for c in kept_after_var if c in X_test.columns]]

    # --- Step 2: Correlation filter ---
    print("\n[3/4] Correlation filtering ...")
    kept_after_corr = correlation_filter(X_train, args.correlation_threshold)
    X_train = X_train[kept_after_corr]
    X_test  = X_test[[c for c in kept_after_corr if c in X_test.columns]]

    # --- Step 3: Mutual information filter ---
    print("\n[4/4] Mutual information filtering ...")
    y_series = pd.Series(y_train, name="target_RUL").fillna(0)
    kept_final = mutual_info_filter(X_train, y_series, args.mi_top_k)
    X_train = X_train[kept_final]
    X_test  = X_test[[c for c in kept_final if c in X_test.columns]]

    print(f"\n  ✅ Final feature count: {len(kept_final)}")

    # --- Save ---
    os.makedirs(args.output_train_features, exist_ok=True)
    os.makedirs(args.output_test_features,  exist_ok=True)
    os.makedirs(args.output_selected_cols,  exist_ok=True)

    train_out = pd.concat([train_df[meta_cols], X_train], axis=1)
    test_out  = pd.concat([test_df[meta_cols],  X_test],  axis=1)

    train_out.to_parquet(os.path.join(args.output_train_features, "data.parquet"), index=False)
    test_out.to_parquet( os.path.join(args.output_test_features,  "data.parquet"), index=False)

    with open(os.path.join(args.output_selected_cols, "selected_features.json"), "w") as f:
        json.dump(kept_final, f, indent=2)

    elapsed = time.time() - start
    print(f"\n✅ reduce_features complete in {elapsed:.1f}s")
    print(f"   Selected features saved → {args.output_selected_cols}/selected_features.json")


if __name__ == "__main__":
    main()