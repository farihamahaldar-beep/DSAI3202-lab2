import argparse
import os
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser(description="Split feature-selected train data into train/validation sets")
    parser.add_argument("--input_features", type=str, required=True,
                        help="Parquet folder — GA-selected train features")
    parser.add_argument("--input_labels",   type=str, required=True,
                        help="Parquet folder — aligned RUL labels")
    parser.add_argument("--output_train",   type=str, required=True,
                        help="Output: training split (features + label)")
    parser.add_argument("--output_val",     type=str, required=True,
                        help="Output: validation split (features + label)")
    parser.add_argument("--val_size",       type=float, default=0.2,
                        help="Fraction of data for validation (default 0.20)")
    parser.add_argument("--random_seed",    type=int,   default=42)
    parser.add_argument("--stratify_bins",  type=int,   default=10,
                        help="Bin count for stratified split on RUL (0 = no stratify)")
    return parser.parse_args()


def load_parquet_folder(folder_path: str) -> pd.DataFrame:
    files = [f for f in os.listdir(folder_path) if f.endswith(".parquet")]
    if not files:
        raise FileNotFoundError(f"No parquet files in {folder_path}")
    return pd.concat(
        [pd.read_parquet(os.path.join(folder_path, f)) for f in files],
        ignore_index=True
    )


def main():
    args = parse_args()
    start = time.time()

    print("=" * 60)
    print("COMPONENT: split_dataset")
    print("=" * 60)

    # --- Load ---
    print("\n[1/2] Loading data ...")
    features_df = load_parquet_folder(args.input_features)
    labels_df   = load_parquet_folder(args.input_labels)

    # Merge features with labels on entity_id
    df = features_df.merge(
        labels_df[["entity_id", "target_RUL"]],
        on="entity_id",
        how="inner"
    )
    df.dropna(subset=["target_RUL"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"  Merged dataset: {len(df)} rows, {len(df.columns)} columns")
    print(f"  RUL range: [{df['target_RUL'].min():.0f}, {df['target_RUL'].max():.0f}]")

    # --- Split ---
    print(f"\n[2/2] Splitting {int((1-args.val_size)*100)}/{int(args.val_size*100)} "
          f"train/val (seed={args.random_seed}) ...")

    stratify_col = None
    if args.stratify_bins > 0:
        # Bin RUL for stratified split (ensures equal distribution of engine health)
        df["_rul_bin"] = pd.cut(
            df["target_RUL"],
            bins=args.stratify_bins,
            labels=False,
            duplicates="drop"
        )
        # Only stratify if each bin has >= 2 samples
        bin_counts = df["_rul_bin"].value_counts()
        if bin_counts.min() >= 2:
            stratify_col = df["_rul_bin"]
            print(f"  Stratifying on {args.stratify_bins} RUL bins")
        else:
            print("  Skipping stratification (some bins have < 2 samples)")
        df.drop(columns=["_rul_bin"], inplace=True)

    train_df, val_df = train_test_split(
        df,
        test_size=args.val_size,
        random_state=args.random_seed,
        stratify=stratify_col,
    )

    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)

    print(f"  Train split: {len(train_df)} rows")
    print(f"  Val   split: {len(val_df)} rows")
    print(f"  Train RUL mean: {train_df['target_RUL'].mean():.1f} ± "
          f"{train_df['target_RUL'].std():.1f}")
    print(f"  Val   RUL mean: {val_df['target_RUL'].mean():.1f} ± "
          f"{val_df['target_RUL'].std():.1f}")

    # --- Save ---
    os.makedirs(args.output_train, exist_ok=True)
    os.makedirs(args.output_val,   exist_ok=True)

    train_df.to_parquet(os.path.join(args.output_train, "data.parquet"), index=False)
    val_df.to_parquet(  os.path.join(args.output_val,   "data.parquet"), index=False)

    elapsed = time.time() - start
    print(f"\n✅ split_dataset complete in {elapsed:.1f}s")
    print(f"   Train → {args.output_train}")
    print(f"   Val   → {args.output_val}")


if __name__ == "__main__":
    main()