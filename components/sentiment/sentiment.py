import argparse
import os
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download required VADER lexicon
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, required=True)
    parser.add_argument("--output_data", type=str, required=True)
    parser.add_argument("--text_column", type=str, default="reviewText")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load data from parquet
    if os.path.isdir(args.input_data):
        input_path = os.path.join(args.input_data, "data.parquet")
    else:
        input_path = args.input_data
    
    df = pd.read_parquet(input_path)
    print(f"Input shape: {df.shape}")
    
    # Ensure text column exists
    if args.text_column not in df.columns:
        raise ValueError(f"Column {args.text_column} not found!")
    
    # Fill empty reviews
    df[args.text_column] = df[args.text_column].fillna('')
    
    # Initialize VADER sentiment analyzer
    sia = SentimentIntensityAnalyzer()
    
    # Extract sentiment features - KEEP SAME INDEX
    sentiment_scores = []
    for text in df[args.text_column]:
        scores = sia.polarity_scores(str(text))
        sentiment_scores.append({
            'sentiment_pos': scores['pos'],
            'sentiment_neg': scores['neg'],
            'sentiment_neu': scores['neu'],
            'sentiment_compound': scores['compound']
        })
    
    # Create dataframe with SAME INDEX as original
    sentiment_df = pd.DataFrame(sentiment_scores, index=df.index)
    
    # Add sentiment features to original dataframe by column, preserving index
    df = pd.concat([df, sentiment_df], axis=1)
    
    print(f"Output shape: {df.shape}")
    print("Successfully created sentiment features.")
    
    # Save the output
    os.makedirs(args.output_data, exist_ok=True)
    df.to_parquet(os.path.join(args.output_data, "data.parquet"), index=False)

if __name__ == "__main__":
    main()