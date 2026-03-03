# Lab 4: Text Feature Engineering with Azure ML

## Overview

This lab implements a complete feature engineering pipeline using Azure Machine Learning to transform raw Amazon Electronics review text into machine learning-ready numerical features. The pipeline processes 300,000+ reviews and generates 528 engineered features across four key dimensions: review length, sentiment, word importance (TF-IDF), and semantic meaning (BERT embeddings). The goal was to transform raw text into high-signal numerical features, merge them into a unified dataset, and register them as a versioned Feature Set in the Azure Machine Learning Feature Store.

## Part A: Dataset Exploration & Validation
Load & Inspect: Verify dataset loads correctly, check row/column counts, schema, and data types to catch issues early.
Verify Data Quality: Check for missing values and malformed reviews in key columns (reviewText, overall, asin).

**Visualizations**
Rating Distribution: Heavily skewed toward 5-star (~17M reviews) vs 1-star (~2M), creating class imbalance requiring class weights in ML models.
Review Length Distribution: Most reviews ~500 characters, but long tail extends to 30,000+. Length alone doesn't predict ratings, but combined with sentiment it's useful.
**Drift-Resistant Sampling**
Problem: Random sampling from 20M reviews over-represents recent years, causing language drift when features fail on older data.
Solution: Stratified sampling by year ensures equal representation across 1999-2014 timespan:
pythondf_with_year = df.withColumn("review_year", F.year(F.from_unixtime(df.reviewTime.cast('long'))))
df_sampled = df_with_year.stat.sampleBy('review_year', fractions={...})
Result: 300K sample representative of full dataset with rating distributions matching original (~43% 5-star).

## Part B: Feature Engineering Components
Each component performs one specific task, runs on Azure ML compute, and outputs parquet files that feed into the next stage.
## Part B: Feature Engineering Components

Each component performs one specific task, runs on Azure ML compute, and outputs parquet files that feed into the pipeline.

<details>
<summary><b>📁 File Structure</b></summary>components/
├── split_dataset/
│   ├── split.py
│   └── component.yml
├── normalize_text/
│   ├── normalize.py
│   └── component.yml
├── review_length/
│   ├── review_length.py
│   └── component.yml
├── sentiment/
│   ├── sentiment.py
│   ├── component.yml
│   └── conda.yml
├── tfidf/
│   ├── tf_idf.py
│   ├── component.yml
│   └── conda.yml
├── semantic_embedding/
│   ├── semantic.py
│   ├── component.yml
│   └── conda.yml
└── merge_features/
├── merge.py
└── component.yml

</details>

### 1. Split Dataset Component

**Purpose:** Prevent data leakage by splitting BEFORE feature fitting.

**Code Explanation:** Two-stage split strategy: 70/30 split (train vs temp), then 50/50 of temp (validation vs test) = exactly 70% train, 15% validation, 15% test. Seed=42 ensures reproducible splits every run. Two stages needed because sklearn doesn't natively do 70/15/15.

**Registration:**
```bash
az ml component create -f components/split_dataset/component.yml
```

**Output:** train/ (210k rows), val/ (45k rows), test/ (45k rows)

**Environment:** Standard sklearn (AzureML-sklearn-1.1-ubuntu20.04-py38-cpu@latest)

---

### 2. Normalize Text Component

**Purpose:** Consistent text preprocessing across all splits to prevent train/test distribution mismatch.

**Code Explanation:** Applies regex patterns to standardize: lowercase letters ("Great" = "great"), remove URLs/numbers (noise), strip punctuation, trim whitespace, filter reviews <10 chars. Identical preprocessing on train/val/test prevents models from seeing different patterns in different splits.

**Registration:**
```bash
az ml component create -f components/normalize_text/component.yml
```

**Applied to:** normalize_train, normalize_val, normalize_test (3 parallel instances)

**Environment:** Standard sklearn

---

### 3. Review Length Features

**Purpose:** Capture review effort and engagement.

**Created:** 
- `review_length_words` (word count)
- `review_length_chars` (character count)

**Code Explanation:** Counts words by splitting on whitespace and counts total characters. Simple but interpretable—1-star reviews averaging 2000 chars vs 5-star averaging 400 chars is a strong signal. Longer reviews = more detailed complaints or recommendations.

**Registration:**
```bash
az ml component create -f components/review_length/component.yml
```

**Input:** Normalized training data | **Output:** review_length features appended

**Environment:** Standard sklearn

---

### 4. Sentiment Features (VADER)

**Purpose:** Extract emotional tone and opinion polarity from review text.

**Created:** 
- `sentiment_pos` (positive word proportion, 0-1)
- `sentiment_neg` (negative word proportion, 0-1)
- `sentiment_neu` (neutral word proportion, 0-1)
- `sentiment_compound` (normalized combined score, -1 to +1)

**Code Explanation:** VADER calculates four sentiment proportions for each review. Pre-trained on social media (tweets, reviews); handles contractions ("don't" → negative), emoji, capitalization emphasis ("GREAT" > "great"), punctuation ("Great!!!" > "Great")—exactly what Amazon reviews contain.

**Registration:**
```bash
az ml component create -f components/sentiment/component.yml
```

**Dependencies:** nltk>=3.6.0, pandas>=1.3.0, pyarrow>=10.0.0

**Critical:** Must include `pyarrow>=10.0.0` for parquet I/O support.

**Environment:** Custom conda (nltk, pandas, pyarrow)

---

### 5. TF-IDF Features

**Purpose:** Capture word frequency and importance using statistical weighting.

**Config:** 
- max_features=100 (top 100 most important words)
- ngram_range=(1,2) (unigrams + bigrams)
- stop_words='english' (remove common filler)

**Created:** ~100 features named `tfidf_<word>` (e.g., `tfidf_great`, `tfidf_poor`)

**Code Explanation:** Vectorizer fit ONLY on training data (critical to prevent data leakage); same vocabulary applied to val/test without learning from them. Bigrams capture negations ("not good" ≠ "good"). Removes stop words ("the", "is") as noise. Top 100 words reduce dimensionality from 10,000+ possible terms.

**Registration:**
```bash
az ml component create -f components/tfidf/component.yml
```

**Data Leakage Prevention:** Fit vectorizer on train_data only; transform val/test without fitting.

**Dependencies:** scikit-learn>=0.24.0, pandas>=1.3.0, numpy>=1.20.0, pyarrow>=10.0.0

**Critical:** Must include `pyarrow>=10.0.0` for parquet output.

**Environment:** Custom conda (scikit-learn, pandas, numpy, pyarrow)

---

### 6. Semantic Embedding Features (BERT)

**Purpose:** Capture contextual meaning and semantic relationships using transformer models.

**Model:** all-MiniLM-L6-v2 (DistilBERT)
- 66% faster than full BERT
- 40% smaller than full BERT
- 95% of BERT performance

**Created:** 384 features `bert_embedding_0` through `bert_embedding_383`

**Code Explanation:** DistilBERT pre-trained on 1 billion+ sentences converts each review into 384-dimensional semantic vector. Synonyms like "great" and "excellent" have similar values. Understands context ("bank" in "river bank" ≠ "bank account"). Why embeddings when we have TF-IDF? Because embeddings understand meaning; TF-IDF only sees word frequencies. 384 dimensions balance capturing nuance against computation cost.

**Registration:**
```bash
az ml component create -f components/semantic_embedding/component.yml
```

**Dependencies:** torch>=2.0.0, transformers>=4.35.0, sentence-transformers>=2.2.2, pandas>=1.5.0, numpy>=1.21.0, pyarrow>=10.0.0

**Critical:** Must include torch, transformers, sentence-transformers, and pyarrow.

**Performance:** Slowest component (~2-3 minutes on cluster). Uses batch processing (batch_size=32) to optimize inference speed.

**Environment:** Custom conda (torch, transformers, sentence-transformers, pyarrow)

---

### 7. Merge Features Component

**Purpose:** Consolidate all engineered features into single dataset for Feature Store registration.

**Code Explanation:** Inner joins all 4 feature outputs on entity keys (asin = product ID, reviewerID = reviewer ID), keeping only rows with complete feature vectors (no missing values). Ensures every row has all features for ML models.

**Registration:**
```bash
az ml component create -f components/merge_features/component.yml
```

**Data Volume:**
- Input: 300k sampled reviews
- After split: 300k total (210k train + 45k val + 45k test)
- Output: 403,409 rows × 528 columns (11 original + 517 engineered)
- Row increase due to duplication in feature joins

**Environment:** Standard sklearn

---

## Feature Engineering Summary

| Component | Purpose | Features | Environment |
|-----------|---------|----------|-------------|
| Split | Prevent data leakage | Train/val/test splits | Standard |
| Normalize | Standardize text | Cleaned text | Standard |
| Review Length | Engagement signal | 2 features | Standard |
| Sentiment (VADER) | Emotional tone | 4 features | Custom |
| TF-IDF | Word importance | ~100 features | Custom |
| BERT | Semantic meaning | 384 features | Custom |
| Merge | Consolidate | 528 columns | Standard |

---

## Common Issues & Fixes

**Missing pyarrow:** Add `pyarrow>=10.0.0` to conda.yml for sentiment, tfidf, semantic_embedding components.

**File path mismatch:** Ensure component.yml command line matches actual Python filename exactly.

**Data leakage in TF-IDF:** Vectorizer must fit on train_data only; use transform (no fit) on val/test.

**Component registration fails:** Increment version number in component.yml to create new version (version: 1 → version: 2).

---
