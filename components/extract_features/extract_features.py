import argparse
import os
import pandas as pd
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters
from tsfresh.utilities.dataframe_functions import impute

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()

def load_data(path):
    """Load data from parquet file in folder"""
    if os.path.isdir(path):
        parquet_path = os.path.join(path, "data.parquet")
        if os.path.exists(parquet_path):
            return pd.read_parquet(parquet_path)
    else:
        if path.endswith('.parquet'):
            return pd.read_parquet(path)
    raise FileNotFoundError(f"Could not find data file in {path}")

def main():
    args = parse_args()
    
    print("Loading preprocessed data...")
    train_df = load_data(args.train_data)
    test_df = load_data(args.test_data)
    
    # Get sensor columns
    sensor_cols = [c for c in train_df.columns if c.startswith('sensor_') or c.startswith('op_setting_')]
    
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    print(f"Sensor columns: {len(sensor_cols)}")
    
    # Extract features using tsfresh
    print("Extracting train features with EfficientFCParameters...")
    train_features = extract_features(
        train_df[['engine_id', 'cycle'] + sensor_cols],
        column_id='engine_id',
        column_sort='cycle',
        default_fc_parameters=EfficientFCParameters(),
        n_jobs=4,
        disable_progressbar=False
    )
    impute(train_features)
    
    print("Extracting test features with EfficientFCParameters...")
    test_features = extract_features(
        test_df[['engine_id', 'cycle'] + sensor_cols],
        column_id='engine_id',
        column_sort='cycle',
        default_fc_parameters=EfficientFCParameters(),
        n_jobs=4,
        disable_progressbar=False
    )
    impute(test_features)
    
    # Compute RUL per cycle
    train_df['RUL'] = train_df.groupby('engine_id')['cycle'].transform('max') - train_df['cycle']
    test_df['RUL'] = test_df.groupby('engine_id')['cycle'].transform('max') - test_df['cycle']
    
    train_target = train_df['RUL'].values
    test_target = test_df.groupby('engine_id')['RUL'].first().values
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save outputs
    train_features.to_parquet(os.path.join(args.output_dir, "train_features.parquet"))
    test_features.to_parquet(os.path.join(args.output_dir, "test_features.parquet"))
    pd.DataFrame({'RUL': train_target}).to_csv(os.path.join(args.output_dir, "train_target.csv"), index=False)
    pd.DataFrame({'RUL': test_target}).to_csv(os.path.join(args.output_dir, "test_target.csv"), index=False)
    
    print(f"✅ Feature extraction complete!")
    print(f"Train features: {train_features.shape}")
    print(f"Test features: {test_features.shape}")

if __name__ == "__main__":
    main()