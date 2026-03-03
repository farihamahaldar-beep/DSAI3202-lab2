import argparse
import os
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--review_length_data", type=str, required=True)
    parser.add_argument("--sentiment_data", type=str, required=True)
    parser.add_argument("--tfidf_data", type=str, required=True)
    parser.add_argument("--semantic_embedding_data", type=str, required=True)
    parser.add_argument("--output_data", type=str, required=True)
    parser.add_argument("--entity_keys", type=str, default="asin,reviewerID")
    return parser.parse_args()

def load_data(path):
    """Load data from parquet file in folder"""
    if os.path.isdir(path):
        parquet_path = os.path.join(path, "data.parquet")
        if os.path.exists(parquet_path):
            return pd.read_parquet(parquet_path)
        # Fallback to CSV
        csv_path = os.path.join(path, "data.csv")
        if os.path.exists(csv_path):
            return pd.read_csv(csv_path)
    else:
        if path.endswith('.parquet'):
            return pd.read_parquet(path)
        elif path.endswith('.csv'):
            return pd.read_csv(path)
    
    raise FileNotFoundError(f"Could not find data file in {path}")

def main():
    args = parse_args()
    
    entity_keys = args.entity_keys.split(',')
    
    print("Loading feature datasets...")
    
    review_length_df = load_data(args.review_length_data)
    sentiment_df = load_data(args.sentiment_data)
    tfidf_df = load_data(args.tfidf_data)
    semantic_embedding_df = load_data(args.semantic_embedding_data)
    
    print(f"Review Length features shape: {review_length_df.shape}")
    print(f"Sentiment features shape: {sentiment_df.shape}")
    print(f"TF-IDF features shape: {tfidf_df.shape}")
    print(f"Semantic Embedding features shape: {semantic_embedding_df.shape}")
    
    # Start with review_length as base dataframe
    merged_df = review_length_df.copy()
    
    # Merge sentiment features
    print(f"\nMerging sentiment features on keys: {entity_keys}")
    merged_df = merged_df.merge(
        sentiment_df,
        on=entity_keys,
        how='inner',
        suffixes=('', '_sentiment')
    )
    print(f"Shape after sentiment merge: {merged_df.shape}")
    
    # Merge TF-IDF features
    print(f"Merging TF-IDF features on keys: {entity_keys}")
    merged_df = merged_df.merge(
        tfidf_df,
        on=entity_keys,
        how='inner',
        suffixes=('', '_tfidf')
    )
    print(f"Shape after TF-IDF merge: {merged_df.shape}")
    
    # Merge semantic embedding features
    print(f"Merging semantic embedding features on keys: {entity_keys}")
    merged_df = merged_df.merge(
        semantic_embedding_df,
        on=entity_keys,
        how='inner',
        suffixes=('', '_embedding')
    )
    print(f"Shape after semantic embedding merge: {merged_df.shape}")
    
    # Create output directory
    os.makedirs(args.output_data, exist_ok=True)
    
    # Save merged dataset as Parquet
    output_path = os.path.join(args.output_data, "data.parquet")
    merged_df.to_parquet(output_path, index=False)
    
    print(f"\nMerge completed successfully!")
    print(f"Merged dataset shape: {merged_df.shape}")
    print(f"Total features: {len(merged_df.columns)}")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    main()