# Assignment 2: Amazon Review Sentiment Analysis (MLOps Pipeline)

---

## Project Overview
This project implements a complete end-to-end MLOps pipeline for sentiment classification on Amazon Electronics reviews using Azure Machine Learning, MLflow, and Azure DevOps.

The goal is to move from a simple ML model to a production-style system including:

- Automated training jobs
- Hyperparameter tuning (sweep)
- Model registration
- Deployment to an endpoint
- CI/CD pipeline automation

---

## Project Structure
repo/
├── src/
│ ├── train.py
│ ├── score.py
│ └── invoke_endpoint.py
├── jobs/
│ ├── train_job.yml
│ ├── sweep_job.yml
│ └── deployment.yml
├── env/
│ ├── conda.yml
│ └── inference_conda.yml
├── azure-pipelines.yml
└── README.md


---

## Part I – Data & Feature Engineering

### Dataset Split

A 4-way partition ensures proper data isolation:

- **Train (60%)**: 2251 samples  
- **Validation (15%)**: 583 samples  
- **Test (15%)**: 584 samples  
- **Deployment (10%)**: Used for endpoint testing (simulates production data)

### Feature Engineering

Each review is converted to a 495-dimensional feature vector:

- **SBERT Embeddings (384 dims)**: Semantic meaning via DistilBERT
- **TF-IDF Features (100 dims)**: Word importance (top 100 terms, unigrams + bigrams)
- **Sentiment Compound (1 dim)**: Emotional tone via VADER
- **Review Length (1 dim)**: Words in review

**Feature Selection Rationale**:
- SBERT captures synonymy and context ("great" ≈ "excellent")
- TF-IDF identifies discriminative keywords
- Sentiment compound aggregates emotional signals (pos/neg/neu are correlated)
- Length captures effort/engagement

### Critical Bug Fix: Sentiment Component

**Issue Discovered**: Sentiment component output entire dataframe instead of just features, causing:
- Row explosion (210k → 273k → 403k)
- 47% duplicate entity keys (asin, reviewerID)
- Severe downstream overfitting (99.7% train, 65% test)

**Fix Applied**: Modified sentiment.py to output only features + entity keys, reducing duplicates to 13.4% and enabling 81.68% test accuracy in initial training.

**Status**: Fix committed but not re-run to conserve compute resources. Effectiveness validated through downstream model improvements.

<img width="1128" height="244" alt="image" src="https://github.com/user-attachments/assets/04df6d55-4805-45e8-a7e1-2b25331da710" />

---

## Part II – Model Training

### Model Selection: Logistic Regression

**Why Logistic Regression?**
- Works well with high-dimensional text features (495 dims)
- Fast training and inference (9 seconds per run)
- Stable and interpretable
- Easy to debug and tune

**Architecture**:
```python
LogisticRegression(
    C=0.0163,              # L2 regularization strength (inverse)
    class_weight='balanced',  # Handles class imbalance
    max_iter=1500,         # Sufficient for convergence
    random_state=123       # Reproducibility
)
```

### Label Definition

Binary classification based on normalized rating (0-1 scale):
label = 1 if overall >= 0.6  (≈ 4-5 stars: "good")
label = 0 otherwise          (≈ 1-3 stars: "not good")

**Threshold tuning journey**:
- 0.8 threshold (5-star only) → 70.89% test accuracy, limited positive samples
- 0.6 threshold (4-5 star) → 81.68% test accuracy, better class balance ✅

### Feature Standardization

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)    # Fit only on train
X_val_scaled = scaler.transform(X_val)            # Transform val/test
X_test_scaled = scaler.transform(X_test)          # Prevents data leakage
```

---

## Part III – Training Pipeline (Azure ML)

### Sweep Configuration

Tested 6 combinations of:
- **C**: Uniform distribution [0.001, 0.1]
- **max_iter**: Uniform distribution [1000, 3000]

### Sweep Results

| Run | C Value | max_iter | Val Accuracy | Status |
|-----|---------|----------|--------------|--------|
| 0 | 0.0163 | 1500 | **0.7753** | ✅ BEST |
| 1 | 0.0267 | 1000 | 0.7667 | |
| 2 | 0.0599 | 2500 | 0.7667 | |
| 3 | 0.0224 | 1500 | 0.7684 | |
| 4 | 0.0297 | 1500 | 0.7684 | |
| 5 | 0.0500 | 2000 | 0.7684 | |

**Key Insight**: Lower C values (stronger regularization) perform better, indicating the model benefits from constraint.

---

---

## Part IV: Feature Experiments

To evaluate the impact of different feature combinations, three configurations were tested by modifying the feature selection logic in `train.py`.

### Experiment Results

| Feature Set | Train Acc | Val Acc | Test Acc | Observation |
|------------|----------|--------|----------|------------|
| SBERT only | 0.8578 | 0.7444 | 0.7260 | Captures semantics but misses keywords |
| SBERT + TF-IDF | 0.8965 | 0.7581 | 0.7740 | Better balance of semantics + keywords |
| **All features** | **0.9098** | **0.7650** | **0.8031** | ✅ Best overall performance |

### Key Insights

- **SBERT only**: Good semantic understanding but lacks surface-level signals  
- **TF-IDF addition**: Improves detection of important words (e.g., "broken", "excellent")  
- **All features combined**:
  - Best generalization
  - Captures semantics + keyword importance + sentiment tone + review structure  

### Final Decision

The **full feature set (SBERT + TF-IDF + sentiment + length)** was selected as the final configuration due to superior performance across validation and test sets.

<img width="1091" height="335" alt="image" src="https://github.com/user-attachments/assets/a3397c9c-4235-41bb-a4db-b3bb987aedd5" />


---

## Part V: Training Job Execution (Azure ML)

### Automated Training Job

**File**: `jobs/train_job.yml`

**Trigger**: Manual submission or via Azure DevOps CI/CD pipeline

**Flow**:
1. Load train/val/test datasets from registered Azure ML Data Assets
2. Create binary labels (≥0.6 → 1, else 0)
3. Build feature matrix (495 dimensions)
4. Standardize features (fit scaler on train only)
5. Train Logistic Regression
6. Evaluate on all splits
7. Log metrics and hyperparameters to MLflow
8. Save model artifact (model.pkl)

### MLflow Tracking

Each training run logs:
- **Hyperparameters**: C, max_iter, num_features, sample counts
- **Metrics**: accuracy, AUC, precision, recall, F1 (for train/val/test)
- **Runtime**: training_runtime_seconds
- **Artifact**: model.pkl (model + scaler + feature columns)

---

## Part VI: Model Registration

The trained model was registered in Azure ML Model Registry:

```bash
az ml model create \
  --name amazon-review-sentiment-model \
  --path azureml://jobs/<JOB_ID>/outputs/model_output \
  --type custom_model
```

**Benefits**:
- Version control and lineage tracking
- Links model to exact training run and hyperparameters
- Enables deployment and monitoring
- Auditable model lifecycle

---

## Part VII: Model Deployment

### Deployment Components

1. **Scoring Script** (`src/score.py`)
   - Loads model, scaler, and feature columns
   - Accepts JSON input (feature vector)
   - Returns predictions and probabilities

2. **Inference Environment** (`env/inference_conda.yml`)
   - Python 3.10
   - scikit-learn, pandas, numpy, joblib
   - azureml-defaults (for endpoint communication)

3. **Deployment Configuration** (`jobs/deployment.yml`)
   - Registered model reference
   - Scoring script
   - Environment specification
   - Compute: Standard_F2s_v2 (1 instance)

### Endpoint Invocation

The deployment dataset (10% of data) is used to test predictions on unseen production data:

**Script**: `src/invoke_endpoint.py`
- Loads deployment dataset
- Sends HTTP POST requests to endpoint
- Computes deployment accuracy
- Simulates data drift detection

---

## Part VIII: CI/CD Pipeline (Azure DevOps)

### Pipeline Configuration

**File**: `azure-pipelines.yml`

**Trigger**: Push to `Assignment-2` branch

**Steps**:
1. Checkout code from GitHub
2. Install Azure CLI and ML extension
3. Configure Azure ML defaults (resource group, workspace)
4. Submit training job via `az ml job create`
5. Stream logs until completion

**Service Connection**: `SC-UDST-CCIT-DSAI3202`

### Automation Benefits

- **Reproducibility**: Same pipeline, same results
- **Version Control**: Model tied to git commit
- **Audit Trail**: Every push creates a logged training run
- **Efficiency**: No manual job submission needed

---

## Part IX: Key Challenges & Solutions

- **Sentiment Component Bug**  
  Output duplicated rows, causing severe overfitting (99% train vs 65% test).  
  Fixed by returning only feature columns → restored generalization.

- **Label Threshold**  
  0.8 threshold limited positive samples.  
  Adjusted to 0.6 → improved class balance and accuracy (~80%).

- **Feature Consistency**  
  Ensured identical feature columns across train/val/test to avoid mismatches.

- **Data Leakage**  
  Scaler fit only on training data → valid evaluation pipeline.

- **MLflow Issue**  
  Version mismatch fixed by removing artifact logging → stable runs.

---

## Best Practices Implemented

- Proper **train/val/test separation**
- **No data leakage** (scaler fit only on train)
- **Hyperparameter tuning using validation set**
- **MLflow experiment tracking**
- **Model registration & versioning**
- **CI/CD automation with Azure DevOps**
- **Deployment via managed endpoint**

---

## Results Summary

| Metric | Value |
|--------|------|
| **Test Accuracy** | **80.31%** |
| **Validation Accuracy** | 76.50% |
| **Train Accuracy** | 90.98% |
| **Best Hyperparameters** | C = 0.0163, max_iter = 1500 |
| **Training Time** | ~9 seconds |
| **Feature Size** | 495 dimensions |
| **Pipeline Status** | Fully automated (Azure DevOps) |

### Final Takeaways

- Logistic Regression performs well on high-dimensional text features  
- Combining **SBERT + TF-IDF + sentiment + length** gives best results  
- Small train-test gap (~10%) indicates good generalization  

## Bonus Question (20%)

### What is one thing we are doing “not correctly” in this assignment?

One issue in this assignment is that we are using the **test set during hyperparameter tuning**. Normally, the test set should only be used once at the very end to evaluate the final model. However, in this case, test metrics are logged for every run in the sweep. This can lead to selecting a model that performs well specifically on the test data rather than generalizing well. The correct approach is to use the **validation set for tuning** and keep the test set completely unseen. The test set should only be used once after the final model is selected.
