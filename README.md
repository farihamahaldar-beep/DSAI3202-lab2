# Assignment 2: Amazon Electronics Review Sentiment Analysis: MLOps Pipeline  

---

## Project Overview
This project implements a complete end-to-end MLOps workflow for sentiment classification on Amazon Electronics reviews. The goal was to transition from a static data engineering pipeline (Lab 4) into a production-ready system involving automated training, hyperparameter optimization, and CI/CD integration. The system is built using Azure Machine Learning, MLflow for experiment tracking, and Azure DevOps for automation.

---

<a name="part-i"></a>
## Part I: Data Engineering & Split Strategy

### Revised 4-Way Split
To simulate a real-world production lifecycle, I updated the splitting logic to include a fourth partition. This ensures the model is evaluated not just on historical data, but on a simulated "live" environment.
* **Train (60%)**: Used for model learning and weight optimization.
* **Validation (15%)**: Used for hyperparameter tuning.
* **Test (15%)**: A true holdout set for offline performance evaluation.
* **Deployment (10%)**: Sourced from the most recent reviews by year to test for model robustness against data drift.

### Feature Pipeline
The pipeline generates a high-dimensional feature matrix (495 features total):
* **SBERT Embeddings (384 dims)**: Captures semantic context using `all-MiniLM-L6-v2`.
* **TF-IDF (100 dims)**: Identifies word-frequency importance (unigrams/bigrams).
* **Sentiment Metrics**: VADER compound scores representing emotional tone.
* **Structural Features**: Review word counts to capture customer effort.

---

<a name="part-ii"></a>
## Part II: Model Development & Feature Construction

### Algorithm Selection
I chose **Logistic Regression** as the primary classifier. It is computationally efficient, handles high-dimensional sparse data (TF-IDF) effectively, and provides interpretable feature weights—making it a stable choice for deployment.

### Label Engineering
A major performance boost came from tuning the "Positive" label threshold. The `overall` rating was normalized to a 0–1 scale:
* **Initial 0.8 Threshold**: (5-stars only) Produced 70.89% accuracy due to limited positive samples.
* **Final 0.6 Threshold**: (4–5 stars) Achieved **81.68% accuracy** with better class balance.

---

<a name="part-iii"></a>
## Part III: Azure ML Pipeline Architecture

### MLflow Integration
Every training run is tracked via MLflow within the Azure ML workspace. This logs:
* **Hyperparameters**: C-value, max iterations, and class weights.
* **Metrics**: Accuracy, AUC, Precision, Recall, and F1-score for all four splits.
* **Artifacts**: The serialized `model.pkl` and the `StandardScaler` (fitted strictly on training data to prevent leakage).

### Hyperparameter Sweep
A sweep job (`sweep_job.yml`) was conducted to optimize the model:
* **Sampling**: Random sampling of the `C` parameter.
* **Insight**: The sweep confirmed that a middle-ground regularization ($C=0.02$) provided the best generalization, preventing the overfitting seen in earlier trials.

---

<a name="part-iv"></a>
## Part IV: CI/CD Automation with Azure DevOps

The training workflow is fully automated through `azure-pipelines.yml`:
* **Trigger**: Every push to the `Assignment-2` branch.
* **Service Connection**: `SC-UDST-CCIT-DSAI3202`.
* **Automation Logic**: The pipeline installs the ML CLI, configures workspace defaults, and submits the training job. I explicitly passed resource group and workspace flags to ensure the CLI extension functioned reliably within the DevOps environment.

---

<a name="part-v"></a>
## Part V: Results & Experimental Observations

### Final Performance Metrics
| Split | Accuracy | AUC | Precision | Recall | F1-Score |
|-------|----------|-----|-----------|--------|----------|
| Train | 87.61% | 95.04% | 97.19% | 87.23% | 91.94% |
| **Test** | **81.68%** | **81.38%** | **92.24%** | **85.25%** | **88.60%** |

### Feature Configuration Tests
| Run | Feature Set | Observation |
|-----|-------------|-------------|
| 1 | SBERT Only | High semantic understanding but missed specific keyword triggers. |
| 2 | SBERT + TF-IDF | Significant boost in discriminative power for technical terms. |
| 3 | **All Features** | Best overall performance by combining sentiment tone and review effort. |

---

<a name="part-vi"></a>
## Part VI: Challenges & Technical Lessons

### The "Sentiment Component" Bug
The most significant challenge was a data-integrity issue in the merge phase. The sentiment component was outputting its entire input dataframe, causing a "row explosion" (210k to 403k rows).
* **Impact**: The model showed 99% training accuracy because it was memorizing duplicates, but only 65% on the test set.
* **Solution**: I refactored the component to output only entity keys and unique features, which restored model generalization.

### Data Standardization
I ensured the `StandardScaler` was fit only on the training split. Applying the same scaler to the validation and test sets is critical in MLOps to prevent data leakage and ensure inference consistency.

---

<a name="bonus"></a>
## Bonus: Evaluation Methodology

**Question: What is one thing we are doing “not correctly” in this assignment?**

In this assignment, we are evaluating the **Test Set** and logging its metrics during every single training trial of the hyperparameter sweep. This violates the **Holdout Principle**. By observing the test set performance for every configuration, we risk selecting a model that "fits" the test set rather than one that generalizes well. The correct approach would be to hide the test set entirely until the best model has been finalized based solely on validation metrics.
