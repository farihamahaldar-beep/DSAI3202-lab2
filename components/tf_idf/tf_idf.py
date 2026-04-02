import argparse
import os
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--deploy_data", type=str, required=True)
    parser.add_argument("--train_out", type=str, required=True)
    parser.add_argument("--val_out", type=str, required=True)
    parser.add_argument("--test_out", type=str, required=True)
    parser.add_argument("--deploy_out", type=str, required=True)
    parser.add_argument("--vectorizer_output", type=str, required=True)
    parser.add_argument("--text_column", type=str, default="reviewText")
    parser.add_argument("--max_features", type=int, default=100)
    parser.add_argument("--ngram_range", type=str, default="1,2")
    return parser.parse_args()

def load_data(path):
    """Load parquet from folder or direct path"""
    if os.path.isdir(path):
        parquet_path = os.path.join(path, "data.parquet")
        if os.path.exists(parquet_path):
            return pd.read_parquet(parquet_path)
    return pd.read_parquet(path)

def apply_tfidf(df, vectorizer, text_column):
    """Apply fitted vectorizer to data"""
    df[text_column] = df[text_column].fillna('')
    tfidf_matrix = vectorizer.transform(df[text_column])
    tfidf_dense = tfidf_matrix.toarray()
    feature_names = vectorizer.get_feature_names_out()
    
    tfidf_df = pd.DataFrame(
        tfidf_dense,
        columns=[f"tfidf_{name}" for name in feature_names]
    )
    
    # Concatenate with original data
    result = pd.concat([df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)
    return result

def main():
    args = parse_args()
    
    # Parse ngram_range
    ngram_parts = args.ngram_range.split(',')
    ngram_range = (int(ngram_parts[0]), int(ngram_parts[1]))
    
    print("Loading training data for vectorizer fitting...")
    train_df = load_data(args.train_data)
    
    # Ensure text column exists
    if args.text_column not in train_df.columns:
        raise ValueError(f"Column {args.text_column} not found in training data!")
    
    train_df[args.text_column] = train_df[args.text_column].fillna('')
    
    print(f"Fitting TF-IDF vectorizer on {len(train_df)} training samples...")
    
    # FIT vectorizer ONLY on training data
    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        stop_words='english',
        ngram_range=ngram_range,
        lowercase=True,
        min_df=1,
        max_df=0.95
    )
    
    vectorizer.fit(train_df[args.text_column])
    
    print(f"Vectorizer fitted with {len(vectorizer.get_feature_names_out())} features")
    
    # Save vectorizer for reproducibility
    os.makedirs(args.vectorizer_output, exist_ok=True)
    vectorizer_path = os.path.join(args.vectorizer_output, "tfidf_vectorizer.pkl")
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"Vectorizer saved to: {vectorizer_path}")
    
    # Apply to TRAIN
    print("Applying vectorizer to training data...")
    train_tfidf = apply_tfidf(train_df, vectorizer, args.text_column)
    os.makedirs(args.train_out, exist_ok=True)
    train_tfidf.to_parquet(os.path.join(args.train_out, "data.parquet"), index=False)
    print(f"Train with TF-IDF: {train_tfidf.shape}")
    
    # Apply to VAL
    print("Applying vectorizer to validation data...")
    val_df = load_data(args.val_data)
    val_tfidf = apply_tfidf(val_df, vectorizer, args.text_column)
    os.makedirs(args.val_out, exist_ok=True)
    val_tfidf.to_parquet(os.path.join(args.val_out, "data.parquet"), index=False)
    print(f"Val with TF-IDF: {val_tfidf.shape}")
    
    # Apply to TEST
    print("Applying vectorizer to test data...")
    test_df = load_data(args.test_data)
    test_tfidf = apply_tfidf(test_df, vectorizer, args.text_column)
    os.makedirs(args.test_out, exist_ok=True)
    test_tfidf.to_parquet(os.path.join(args.test_out, "data.parquet"), index=False)
    print(f"Test with TF-IDF: {test_tfidf.shape}")
    
    # Apply to DEPLOY
    print("Applying vectorizer to deployment data...")
    deploy_df = load_data(args.deploy_data)
    deploy_tfidf = apply_tfidf(deploy_df, vectorizer, args.text_column)
    os.makedirs(args.deploy_out, exist_ok=True)
    deploy_tfidf.to_parquet(os.path.join(args.deploy_out, "data.parquet"), index=False)
    print(f"Deploy with TF-IDF: {deploy_tfidf.shape}")
    
    print("\n✓ TF-IDF successfully applied to all splits!")

if __name__ == "__main__":
    main()