import argparse
import pandas as pd
import os
import numpy as np
from sentence_transformers import SentenceTransformer

def main():
    parser = argparse.ArgumentParser(description="Semantic Embedding Feature Extraction Component")
    parser.add_argument("--input_data", type=str, required=True, help="Path to input data")
    parser.add_argument("--output_data", type=str, required=True, help="Path to output data")
    parser.add_argument("--text_column", type=str, default="review_text", help="Name of text column")
    parser.add_argument("--model_name", type=str, default="sentence-transformers/distilbert-base-uncased-mean-tokens", 
                        help="Hugging Face model name for embeddings")
    
    args = parser.parse_args()
    
    # Load data
    df = pd.read_csv(args.input_data)
    
    # Handle NaN values in text column
    df[args.text_column] = df[args.text_column].fillna("")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_data, exist_ok=True)
    
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
    
    # Combine with original data
    result_df = pd.concat([df.reset_index(drop=True), embedding_df.reset_index(drop=True)], axis=1)
    
    # Save output
    output_path = os.path.join(args.output_data, "data.csv")
    result_df.to_csv(output_path, index=False)
    
    print(f"Semantic embeddings extracted successfully!")
    print(f"Output saved to {output_path}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Total features added: {embedding_dim}")
    print(f"Embeddings shape: {embeddings.shape}")

if __name__ == "__main__":
    main()
