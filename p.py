import numpy as np
import pandas as pd
df = pd.read_parquet("data (1).parquet")
print(f"Total rows: {len(df)}")
print(f"Unique (asin, reviewerID) pairs: {df.groupby(['asin', 'reviewerID']).ngroups}")
print(f"Duplicates: {len(df) - df.groupby(['asin', 'reviewerID']).ngroups}")