import argparse
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, required=True)
    parser.add_argument("--output_data", type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load the dataset
    # Note: Using parquet as it is standard for Azure ML Feature Store labs
    df = pd.read_parquet(args.input_data)
    
    # Identify numeric columns to normalize
    # (Excludes index columns like 'asin' or 'reviewerID')
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    if len(numeric_cols) > 0:
        scaler = MinMaxScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        print(f"Normalized columns: {list(numeric_cols)}")
    else:
        print("No numeric columns found to normalize.")

    # Create output directory and save
    os.makedirs(args.output_data, exist_ok=True)
    df.to_parquet(os.path.join(args.output_data, "data.parquet"))
    
    print(f"Successfully saved normalized data to {args.output_data}")

if __name__ == "__main__":
    main()