import argparse
import pandas as pd
import os
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download required VADER lexicon
nltk.download('vader_lexicon', quiet=True)

def main():
    parser = argparse.ArgumentParser(description="Sentiment Feature Extraction Component")
    parser.add_argument("--input_data", type=str, required=True, help="Path to input data")
    parser.add_argument("--output_data", type=str, required=True, help="Path to output data")
    parser.add_argument("--text_column", type=str, default="review_text", help="Name of text column")
    
    args = parser.parse_args()
    
    # Load data
    df = pd.read_csv(args.input_data)
    
    # Initialize VADER sentiment analyzer
    sia = SentimentIntensityAnalyzer()
    
    # Extract sentiment features
    sentiment_scores = []
    
    for text in df[args.text_column]:
        if pd.isna(text):
            # Handle NaN values
            sentiment_scores.append({
                'sentiment_pos': 0.0,
                'sentiment_neg': 0.0,
                'sentiment_neu': 0.0,
                'sentiment_compound': 0.0
            })
        else:
            # Get sentiment scores
            scores = sia.polarity_scores(str(text))
            sentiment_scores.append({
                'sentiment_pos': scores['pos'],
                'sentiment_neg': scores['neg'],
                'sentiment_neu': scores['neu'],
                'sentiment_compound': scores['compound']
            })
    
    # Create dataframe from sentiment scores
    sentiment_df = pd.DataFrame(sentiment_scores)
    
    # Combine with original data
    result_df = pd.concat([df, sentiment_df], axis=1)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_data, exist_ok=True)
    
    # Save output
    output_path = os.path.join(args.output_data, "data.csv")
    result_df.to_csv(output_path, index=False)
    
    print(f"Sentiment features extracted successfully!")
    print(f"Output saved to {output_path}")
    print(f"Sentiment features added: sentiment_pos, sentiment_neg, sentiment_neu, sentiment_compound")

if __name__ == "__main__":
    main()
