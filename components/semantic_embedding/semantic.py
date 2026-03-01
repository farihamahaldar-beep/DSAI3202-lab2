import argparse
import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, required=True)
    parser.add_argument("--output_data", type=str, required=True)
    parser.add_argument("--text_column", type=str, default="reviewText")
    parser.add_argument("--model_name", type=str, default="sentence-transformers/distilbert-base-uncased-mean-tokens")
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
    
    # Load pre-trained model
    print(f"Loading model: {args.model_name}")
    model = SentenceTransformer(args.model_name)
    
    # Extract embeddings
    print(f"Extracting embeddings for {len(df)} reviews...")
    embeddings = model.encode(df[args.text_column].tolist(), show_progress_bar=True)
    
    # Get embedding dimension
    embedding_dim = embeddings.shape[1]
    
    # Create dataframe from embeddings
    embedding_df = pd.DataFrame(
        embeddings,
        columns=[f"bert_embedding_{i}" for i in range(embedding_dim)]
    )
    
    # Add embeddings to original dataframe
    df = pd.concat([df.reset_index(drop=True), embedding_df.reset_index(drop=True)], axis=1)
    
    # Save the output
    os.makedirs(args.output_data, exist_ok=True)
    df.to_parquet(os.path.join(args.output_data, "data.parquet"), index=False)
    
    print(f"Semantic embeddings extracted successfully!")
    print(f"Embedding dimension: {embedding_dim}")

if __name__ == "__main__":
    main()
