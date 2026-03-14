import argparse
import os
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_features", type=str, required=True)
    parser.add_argument("--train_target", type=str, required=True)
    parser.add_argument("--test_features", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()

def load_data(path):
    """Load data from parquet file in folder"""
    if os.path.isdir(path):
        parquet_path = os.path.join(path, "train_features.parquet")
        if os.path.exists(parquet_path):
            return pd.read_parquet(parquet_path)
    else:
        if path.endswith('.parquet'):
            return pd.read_parquet(path)
    raise FileNotFoundError(f"Could not find data file in {path}")

def load_target(path):
    """Load target CSV"""
    if os.path.isdir(path):
        csv_path = os.path.join(path, "train_target.csv")
        if os.path.exists(csv_path):
            return pd.read_csv(csv_path)['RUL'].values
    else:
        if path.endswith('.csv'):
            return pd.read_csv(path)['RUL'].values
    raise FileNotFoundError(f"Could not find target file in {path}")

def main():
    args = parse_args()
    
    print("Loading features and target...")
    train_features = load_data(args.train_features)
    test_features = load_data(args.test_features.replace("train_features", "test_features"))
    train_target = load_target(args.train_target)
    
    print(f"Original features: {train_features.shape[1]}")
    
    # Step 1: Variance threshold
    print("Applying variance threshold...")
    var_filter = VarianceThreshold(threshold=0.01)
    X_train_var = var_filter.fit_transform(train_features)
    selected_var = [train_features.columns[i] for i, keep in enumerate(var_filter.get_support()) if keep]
    print(f"After variance: {len(selected_var)} features")
    
    # Step 2: Correlation filtering
    print("Removing correlated features...")
    X_train_var_df = pd.DataFrame(X_train_var, columns=selected_var)
    corr = X_train_var_df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_corr = [col for col in upper.columns if any(upper[col] > 0.95)]
    selected_corr = [f for f in selected_var if f not in drop_corr]
    X_train_filtered = X_train_var_df.drop(columns=drop_corr)
    print(f"After correlation: {len(selected_corr)} features")
    
    # Step 3: Mutual information
    print("Computing mutual information...")
    mi_scores = mutual_info_regression(X_train_filtered, train_target, random_state=42)
    mi_df = pd.DataFrame({'feature': selected_corr, 'score': mi_scores}).sort_values('score', ascending=False)
    
    mi_threshold = mi_df['score'].quantile(0.25)
    selected_mi = mi_df[mi_df['score'] > mi_threshold]['feature'].tolist()
    print(f"After MI: {len(selected_mi)} features")
    
    # Apply filters to both train and test
    X_train_filtered = train_features[selected_mi]
    X_test_filtered = test_features[selected_mi]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save outputs
    X_train_filtered.to_parquet(os.path.join(args.output_dir, "train_features_filtered.parquet"))
    X_test_filtered.to_parquet(os.path.join(args.output_dir, "test_features_filtered.parquet"))
    pd.DataFrame({'feature': selected_mi}).to_csv(os.path.join(args.output_dir, "selected_features.csv"), index=False)
    
    print(f"✅ Filter selection complete!")
    print(f"Final features: {len(selected_mi)}")

if __name__ == "__main__":
    main()