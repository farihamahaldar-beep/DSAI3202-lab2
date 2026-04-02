import pandas as pd

df = pd.read_parquet("data (1).parquet")  # Your merged train data

# Check feature statistics
print("SBERT columns:")
sbert = [col for col in df.columns if col.startswith('bert_embedding_')]
print(f"  Count: {len(sbert)}")
print(f"  Mean: {df[sbert].mean().mean():.4f}")
print(f"  Std: {df[sbert].std().mean():.4f}")

print("\nTF-IDF columns:")
tfidf = [col for col in df.columns if col.startswith('tfidf_')]
print(f"  Count: {len(tfidf)}")
print(f"  Sparsity: {(df[tfidf] == 0).sum().sum() / (df[tfidf].shape[0] * df[tfidf].shape[1]) * 100:.1f}%")

print("\nOverall rating distribution:")
print(df['overall'].value_counts().sort_index())

print("\nClass balance (label):")
print(df['label'].value_counts())