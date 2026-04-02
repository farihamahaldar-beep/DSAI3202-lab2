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
    parser.add_argument(
        "--model_name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Handle folder or direct path
    if os.path.isdir(args.input_data):
        input_path = os.path.join(args.input_data, "data.parquet")
    else:
        input_path = args.input_data
    
    df = pd.read_parquet(input_path)
    print(f"Input shape: {df.shape}")

    if args.text_column not in df.columns:
        raise ValueError(f"Column {args.text_column} not found!")

    df[args.text_column] = df[args.text_column].fillna("")

    print(f"Loading model: {args.model_name}")
    model = SentenceTransformer(args.model_name)

    print(f"Extracting embeddings for {len(df)} reviews...")
    embeddings = model.encode(
        df[args.text_column].tolist(),
        batch_size=32,
        show_progress_bar=True
    )

    embedding_dim = embeddings.shape[1]

    # CREATE DATAFRAME WITH SAME INDEX AS ORIGINAL
    embedding_df = pd.DataFrame(
        embeddings,
        columns=[f"bert_embedding_{i}" for i in range(embedding_dim)],
        index=df.index  # ← KEY FIX
    )

    # Concatenate preserving index
    df = pd.concat([df, embedding_df], axis=1)

    os.makedirs(args.output_data, exist_ok=True)
    df.to_parquet(
        os.path.join(args.output_data, "data.parquet"),
        index=False
    )

    print("Semantic embeddings extracted successfully!")
    print(f"Output shape: {df.shape}")
    print(f"Embedding dimension: {embedding_dim}")


if __name__ == "__main__":
    main()