import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_features", type=str, required=True)
    parser.add_argument("--train_target", type=str, required=True)
    parser.add_argument("--test_features", type=str, required=True)
    parser.add_argument("--test_target", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def load_data(path):
    """Load parquet file"""
    if os.path.isdir(path):
        parquet_path = os.path.join(path, "train_features_selected.parquet")
        if os.path.exists(parquet_path):
            return pd.read_parquet(parquet_path)
    raise FileNotFoundError(f"Could not find data in {path}")

def load_target(path):
    """Load target CSV"""
    if os.path.isdir(path):
        csv_path = os.path.join(path, "train_target.csv")
        if os.path.exists(csv_path):
            return pd.read_csv(csv_path)['RUL'].values
    raise FileNotFoundError(f"Could not find target in {path}")

def main():
    args = parse_args()
    
    print("Loading GA-selected training features and targets...")
    X_train = load_data(args.train_features)
    y_train = load_target(args.train_target)
    
    print("Loading test features and targets...")
    X_test = load_data(args.test_features.replace("train_features", "test_features"))
    y_test = load_target(args.test_target.replace("train_target", "test_target"))
    
    print(f"Train features: {X_train.shape}, Test features: {X_test.shape}")
    
    # Split training data into train and validation
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train,
        test_size=(1 - args.train_ratio),
        random_state=args.seed,
        shuffle=True
    )
    
    print(f"Train: {X_train_split.shape[0]} samples")
    print(f"Validation: {X_val.shape[0]} samples")
    print(f"Test: {X_test.shape[0]} samples")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save splits
    X_train_split.to_parquet(os.path.join(args.output_dir, "train_features.parquet"))
    X_val.to_parquet(os.path.join(args.output_dir, "val_features.parquet"))
    X_test.to_parquet(os.path.join(args.output_dir, "test_features.parquet"))
    
    pd.DataFrame({'RUL': y_train_split}).to_csv(os.path.join(args.output_dir, "train_target.csv"), index=False)
    pd.DataFrame({'RUL': y_val}).to_csv(os.path.join(args.output_dir, "val_target.csv"), index=False)
    pd.DataFrame({'RUL': y_test}).to_csv(os.path.join(args.output_dir, "test_target.csv"), index=False)
    
    print(f"✅ Data split complete!")

if __name__ == "__main__":
    main()