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
    
    # Extract sentiment features ONLY
    sentiment_scores = []
    for text in df[args.text_column]:
        scores = sia.polarity_scores(str(text))
        sentiment_scores.append({
            'sentiment_pos': scores['pos'],
            'sentiment_neg': scores['neg'],
            'sentiment_neu': scores['neu'],
            'sentiment_compound': scores['compound']
        })
    
    # Create dataframe with sentiment features ONLY (same index as input)
    sentiment_df = pd.DataFrame(sentiment_scores, index=df.index)
    
    # Add entity keys back (asin, reviewerID needed for merging)
    if 'asin' in df.columns:
        sentiment_df['asin'] = df['asin'].values
    if 'reviewerID' in df.columns:
        sentiment_df['reviewerID'] = df['reviewerID'].values
    
    # Add label if it exists (preserve for downstream)
    if 'overall' in df.columns:
        sentiment_df['overall'] = df['overall'].values
    
    print(f"Output shape: {sentiment_df.shape}")
    print(f"Columns: {list(sentiment_df.columns)}")
    print("Successfully created sentiment features.")
    
    # Save only sentiment features + entity keys
    os.makedirs(args.output_data, exist_ok=True)
    sentiment_df.to_parquet(os.path.join(args.output_data, "data.parquet"), index=False)

if __name__ == "__main__":
    main()