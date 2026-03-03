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
    input_path = os.path.join(args.input_data, "data.parquet")
    df = pd.read_parquet(input_path)
    # Ensure text column exists
    if args.text_column in df.columns:
        # Fill empty reviews
        df[args.text_column] = df[args.text_column].fillna('')
        
        # Initialize VADER sentiment analyzer
        sia = SentimentIntensityAnalyzer()
        
        # Extract sentiment features
        sentiment_scores = []
        for text in df[args.text_column]:
            scores = sia.polarity_scores(str(text))
            sentiment_scores.append({
                'sentiment_pos': scores['pos'],
                'sentiment_neg': scores['neg'],
                'sentiment_neu': scores['neu'],
                'sentiment_compound': scores['compound']
            })
        
        # Create dataframe from sentiment scores
        sentiment_df = pd.DataFrame(sentiment_scores)
        
        # Add sentiment features to original dataframe
        df = pd.concat([df, sentiment_df], axis=1)
        
        print("Successfully created sentiment features.")
    else:
        print(f"Warning: Column {args.text_column} not found!")
    
    # Save the output
    os.makedirs(args.output_data, exist_ok=True)
    df.to_parquet(os.path.join(args.output_data, "data.parquet"), index=False)

if __name__ == "__main__":
    main()
