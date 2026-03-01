import argparse
import pandas as pd
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="TF-IDF Feature Extraction Component")
    parser.add_argument("--input_data", type=str, required=True, help="Path to input data")
    parser.add_argument("--output_data", type=str, required=True, help="Path to output data")
    parser.add_argument("--text_column", type=str, default="review_text", help="Name of text column")
    parser.add_argument("--max_features", type=int, default=100, help="Maximum number of features")
    parser.add_argument("--ngram_range", type=str, default="1,2", help="N-gram range (e.g., '1,2' for unigrams and bigrams)")
    parser.add_argument("--is_train", type=str, default="true", help="Whether this is training data (true/false)")
    parser.add_argument("--vectorizer_path", type=str, default=None, help="Path to pre-fitted vectorizer (for validation/test data)")
    
    args = parser.parse_args()
    
    # Parse ngram_range
    ngram_parts = args.ngram_range.split(',')
    ngram_range = (int(ngram_parts[0]), int(ngram_parts[1]))
    
    # Load data
    df = pd.read_csv(args.input_data)
    
    # Handle NaN values in text column
    df[args.text_column] = df[args.text_column].fillna("")
    
    # Convert is_train string to boolean
    is_train = args.is_train.lower() == "true"
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_data, exist_ok=True)
    
    if is_train:
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
        
        # Save vectorizer for later use
        vectorizer_path = os.path.join(args.output_data, "tfidf_vectorizer.pkl")
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(vectorizer, f)
        
        print(f"TF-IDF vectorizer fitted and saved to {vectorizer_path}")
        
    else:
        # Load pre-fitted vectorizer
        if args.vectorizer_path is None:
            raise ValueError("vectorizer_path must be provided for validation/test data")
        
        with open(args.vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        
        # Transform using existing vectorizer
        tfidf_matrix = vectorizer.transform(df[args.text_column])
        
        print(f"Data transformed using existing vectorizer from {args.vectorizer_path}")
    
    # Convert sparse matrix to dense and create feature names
    tfidf_dense = tfidf_matrix.toarray()
    feature_names = vectorizer.get_feature_names_out()
    
    # Create dataframe from TF-IDF matrix
    tfidf_df = pd.DataFrame(
        tfidf_dense,
        columns=[f"tfidf_{name}" for name in feature_names]
    )
    
    # Combine with original data
    result_df = pd.concat([df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)
    
    # Save output
    output_path = os.path.join(args.output_data, "data.csv")
    result_df.to_csv(output_path, index=False)
    
    print(f"TF-IDF features extracted successfully!")
    print(f"Output saved to {output_path}")
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"Number of features: {len(feature_names)}")
    print(f"Feature names (first 10): {feature_names[:10]}")

if __name__ == "__main__":
    main()
