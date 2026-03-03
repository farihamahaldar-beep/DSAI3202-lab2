import argparse
import os
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, required=True)
    parser.add_argument("--output_data", type=str, required=True)
    parser.add_argument("--vectorizer_output", type=str, required=True)
    parser.add_argument("--text_column", type=str, default="reviewText")
    parser.add_argument("--max_features", type=int, default=100)
    parser.add_argument("--ngram_range", type=str, default="1,2")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load data from parquet
    df = pd.read_parquet(args.input_data)
    
    # Ensure text column exists
    if args.text_column not in df.columns:
        print(f"Warning: Column {args.text_column} not found!")
        return
    
    # Fill empty reviews
    df[args.text_column] = df[args.text_column].fillna('')
    
    # Parse ngram_range
    ngram_parts = args.ngram_range.split(',')
    ngram_range = (int(ngram_parts[0]), int(ngram_parts[1]))
    
    # Fit TF-IDF vectorizer on training data
    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        stop_words='english',
        ngram_range=ngram_range,
        lowercase=True,
        min_df=1,
        max_df=0.95
    )
    
    # Fit and transform
    tfidf_matrix = vectorizer.fit_transform(df[args.text_column])
    
    # Save vectorizer for later use on validation/test
    os.makedirs(args.vectorizer_output, exist_ok=True)
    vectorizer_path = os.path.join(args.vectorizer_output, "tfidf_vectorizer.pkl")
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    # Convert sparse matrix to dense
    tfidf_dense = tfidf_matrix.toarray()
    feature_names = vectorizer.get_feature_names_out()
    
    # Create dataframe from TF-IDF matrix
    tfidf_df = pd.DataFrame(
        tfidf_dense,
        columns=[f"tfidf_{name}" for name in feature_names]
    )
    
    # Add TF-IDF features to original dataframe
    df = pd.concat([df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)
    
    # Save the output
    os.makedirs(args.output_data, exist_ok=True)
    df.to_parquet(os.path.join(args.output_data, "data.parquet"), index=False)
    
    print(f"TF-IDF features created successfully!")
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"Vectorizer saved for validation/test splits")

if __name__ == "__main__":
    main()