import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--deploy_ratio", type=float, default=0.10)
    parser.add_argument("--train_out", type=str, required=True)
    parser.add_argument("--val_out", type=str, required=True)
    parser.add_argument("--test_out", type=str, required=True)
    parser.add_argument("--deploy_out", type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load dataset - handle both folder and direct parquet
    if os.path.isdir(args.data):
        files = [f for f in os.listdir(args.data) if f.endswith('.parquet')]
        if not files:
            raise FileNotFoundError(f"No parquet file found in {args.data}")
        input_path = os.path.join(args.data, files[0])
    else:
        input_path = args.data
    
    print(f"Reading dataset from: {input_path}")
    df = pd.read_parquet(input_path)
    print(f"Total rows: {len(df)}")
    
    # Sort by review_year so deployment comes from most recent data
    df = df.sort_values('review_year').reset_index(drop=True)
    
    # Extract most recent 10% as deployment split
    split_idx = int(len(df) * (1 - args.deploy_ratio))
    df_main = df.iloc[:split_idx]
    df_deploy = df.iloc[split_idx:]
    
    print(f"Deployment split (most recent): {len(df_deploy)} rows")
    print(f"Remaining data for train/val/test: {len(df_main)} rows")
    
    # Split remaining 90% into train/val/test
    # test_size=0.333 means we take 33.3% of df_main for (val+test)
    # This leaves 66.7% for train, but we want 60% of original
    # So: (90% * 66.7%) / 90% = 60% ✓
    df_train, df_temp = train_test_split(
        df_main,
        test_size=1 - (args.train_ratio / (1 - args.deploy_ratio)),
        random_state=args.seed,
        shuffle=True
    )
    
    # Split the temp (val+test) equally: 50/50
    df_val, df_test = train_test_split(
        df_temp,
        test_size=0.5,
        random_state=args.seed,
        shuffle=True
    )
    
    # Write outputs
    for path, data, name in [
        (args.train_out, df_train, "Train"),
        (args.val_out, df_val, "Validation"),
        (args.test_out, df_test, "Test"),
        (args.deploy_out, df_deploy, "Deployment"),
    ]:
        os.makedirs(path, exist_ok=True)
        data.to_parquet(os.path.join(path, "data.parquet"), index=False)
        print(f"{name} rows: {len(data)} ({100*len(data)/len(df):.1f}%)")
    
    # Sanity check
    total = len(df_train) + len(df_val) + len(df_test) + len(df_deploy)
    assert total == len(df), f"Row count mismatch: {total} vs {len(df)}"
    print(f"\n✓ All splits created successfully. Total: {total} rows")

if __name__ == "__main__":
    main()