import argparse
import os
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, required=True)
    parser.add_argument("--output_data", type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load the data
    df = pd.read_parquet(args.input_data)
    
    # Ensure the review text column exists (usually called 'reviewText' in Amazon data)
    text_col = 'reviewText' 
    
    if text_col in df.columns:
        # Fill empty reviews to avoid errors
        df[text_col] = df[text_col].fillna('')
        
        # 1. Number of words
        df['review_length_words'] = df[text_col].apply(lambda x: len(str(x).split()))
        
        # 2. Number of characters
        df['review_length_chars'] = df[text_col].apply(lambda x: len(str(x)))
        
        print("Successfully created length features.")
    else:
        print(f"Warning: Column {text_col} not found!")

    # Save the output
    os.makedirs(args.output_data, exist_ok=True)
    df.to_parquet(os.path.join(args.output_data, "data.parquet"))

if __name__ == "__main__":
    main()