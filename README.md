# Lab 5 – Scalable Feature Extraction and Selection for Predictive Maintenance

## Overview

This lab builds a scalable end-to-end machine learning pipeline for **Remaining Useful Life (RUL) prediction** of aircraft turbofan engines using the NASA C-MAPSS FD001 dataset. The goal is to predict how many cycles remain before an engine fails based on sensor readings, which is a core problem in predictive maintenance.

The pipeline follows a **medallion architecture** (Bronze → Silver → Gold) on Azure Blob Storage, uses **Azure Databricks** for ETL processing, **tsfresh** for automated time-series feature extraction using rolling windows, a **DEAP genetic algorithm** for optimal feature subset selection, and an **Azure ML Pipeline** with 5 modular command components for scalable and reproducible model training and evaluation.

---

## Part I – Medallion Architecture and Databricks ETL

### 1. Storage Setup and Medallion Architecture

Data was organised across three layers in Azure Blob Storage, following the medallion architecture pattern where each layer represents a progressively cleaner and more enriched version of the data:

- **Bronze layer** (`raw/FD001/`) — raw `.txt` files exactly as received from NASA, never modified. This ensures we can always trace back to the original source.
- **Silver layer** (`processed/FD001/`) — cleaned, correctly typed, and validated data written as Parquet. Column names are assigned, types are cast, and constant sensors are removed.
- **Gold layer** (`curated/FD001/`) — fully normalised, RUL-labelled, tsfresh-ready data for ML consumption.

This separation means that if any downstream step needs to be re-run or changed, only the relevant layer needs to be reprocessed without touching the earlier ones.

---

### 2. Bronze → Silver (Notebook 05)

The raw C-MAPSS files have no column headers and contain trailing empty columns. The first Databricks notebook reads the raw training and test files, assigns all 26 correct column names, casts `engine_id` and `cycle` to integers and all sensor and operational setting columns to floats, and drops the two trailing empty columns.

**Constant sensor removal:** Seven sensors (`sensor_1`, `sensor_5`, `sensor_6`, `sensor_10`, `sensor_16`, `sensor_18`, `sensor_19`) were found to have zero variance across all engines and cycles. These carry no information for RUL prediction and are removed at this stage to reduce noise and downstream computation.

Output written to the Silver layer as Parquet:
- Train: 20,631 rows, 19 columns
- Test: 13,096 rows, 19 columns

---

### 3. Silver → Gold (Notebook 06)

The second notebook enriches the silver data with two key transformations:

**RUL Computation:** For each engine, RUL at each cycle is computed as `max_cycle − current_cycle`, giving the ground truth label for how many cycles remain before failure.

**RUL Clipping at 125:** Raw RUL values can be very large for engines that ran a long time. However, turbofan sensors only show measurable degradation in approximately the last 125 cycles. Clipping at 125 implements the standard **piecewise linear degradation assumption** used across C-MAPSS benchmarks — it focuses the model on the degradation phase and ignores the early healthy phase where all sensors look identical regardless of engine health.

The gold layer writes three outputs:

- `curated/FD001/train_tsfresh_ready` — sensor features for all 20,631 training rows
- `curated/FD001/test_tsfresh_ready` — sensor features for all 13,096 test rows
- `curated/FD001/train_rul_labels` — per-cycle RUL labels (`engine_id`, `cycle`, `target_RUL`)

---

## Part II – Time-Series Exploration and Validation

### Sensor Behaviour

Several sensors show a clear monotonic trend as engines approach failure. `sensor_2` and `sensor_4` trend upward while `sensor_7` trends downward, confirming they carry strong degradation signal. Several sensor pairs exceed the 0.95 correlation threshold, confirming that redundant features exist in the dataset and justifying the correlation filter step in the pipeline.

---

## Part III – Azure ML Pipeline

### Repository Structure

```
LAB5/
├── databricks/
│   ├── 05_load_and_preprocess_turbofan.ipynb
│   └── 06_feature_extraction_tsfresh.ipynb
├── components/
│   ├── extract_features/
│   │   ├── component.yml
│   │   ├── conda.yml
│   │   └── extract_features.py
│   ├── reduce_features/
│   │   ├── component.yml
│   │   ├── conda.yml
│   │   └── reduce_features.py
│   ├── genetic_algorithm/
│   │   ├── component.yml
│   │   ├── conda.yml
│   │   └── genetic_algorithm.py
│   ├── split_dataset/
│   │   ├── component.yml
│   │   ├── conda.yml
│   │   └── split_dataset.py
│   └── train_evaluate/
│       ├── component.yml
│       ├── conda.yml
│       └── train_evaluate.py
├── config/
│   └── config.yaml
├── pipelines/
│   └── feature_pipeline.yml
└── README.md
```

---

### Registering Components

Each component is registered individually in Azure ML before submitting the pipeline:

```powershell
az ml component create --file components/extract_features/component.yml
az ml component create --file components/reduce_features/component.yml
az ml component create --file components/genetic_algorithm/component.yml
az ml component create --file components/split_dataset/component.yml
az ml component create --file components/train_evaluate/component.yml
```

---

### Pipeline Components — What Each Step Does

#### Extract Features (`extract_features`)

Uses tsfresh to extract statistical time-series features from the sensor data using a **rolling window approach**.

**Why rolling windows?** A naive approach of extracting one feature vector per engine produces only 100 rows. Due to RUL clipping at 125, nearly all engines get the same label for most of their life, leaving almost no variation for the model to learn from. The rolling window approach extracts one feature vector per `(engine, cycle)` pair using the most recent 30 cycles as the input window. This produces thousands of rows with properly varying RUL labels from 0 to 125 at each time step.

`EfficientFCParameters` is used to extract a comprehensive set of statistical features per sensor per window.

| Parameter | Value |
|---|---|
| Feature set | `EfficientFCParameters` |
| Window size | 30 cycles |

---

#### Filter-Based Feature Reduction (`reduce_features`)

Three sequential filters reduce the feature space before the expensive genetic algorithm runs. Running filters first shrinks the search space significantly, making the GA faster and reducing the risk of selecting noise features.

| Step | Method | Threshold | Purpose |
|---|---|---|---|
| 1 | Variance threshold | < 0.01 | Remove near-constant features |
| 2 | Correlation filter | > 0.95 | Remove redundant duplicate features |
| 3 | Mutual information | Top 100 | Keep features most informative about RUL |

---

#### Genetic Algorithm Feature Selection (`genetic_algorithm`)

Uses **DEAP** (Distributed Evolutionary Algorithms in Python) to search for the optimal feature subset. Each individual in the population is a binary chromosome where `1` = include feature and `0` = exclude. Fitness is evaluated using 3-fold cross-validated RMSE with a lightweight RandomForest to balance evaluation speed with accuracy.

The GA is preferred over purely filter-based methods because it can discover **feature combinations** that work well together, not just features that are individually strong.

| Parameter | Value |
|---|---|
| Population size | 50 |
| Generations | 30 |
| Crossover probability | 0.7 |
| Mutation probability | 0.2 |
| Selection | Tournament (size 3) |
| Fitness | 3-fold CV RMSE − α × feature ratio |

---

#### Split Dataset (`split_dataset`)

Splits the GA-selected training features into 80% training and 20% validation with a fixed random seed. Stratification bins RUL values to ensure equal distribution of engine health states across both splits. The split occurs after all feature selection steps to ensure no validation data influenced which features were selected or how they were extracted.

---

#### Train and Evaluate (`train_evaluate`)

Trains a **RandomForest Regressor** on the training split and evaluates on the validation split. The model is saved via joblib, predictions are saved as CSV, and all metrics are written to `metrics.json`.

| Parameter | Value |
|---|---|
| Model | RandomForest Regressor |
| n_estimators | 200 |
| max_depth | unlimited |
| OOB score | enabled |

Evaluation includes RMSE, MAE, R², the NASA asymmetric scoring function (which penalises late predictions more heavily than early ones), and feature importances.

---

### Running the Pipeline

Submit the full pipeline from the `LAB5/` root:

```powershell
az ml job create --file pipelines/feature_pipeline.yml
```

Stream logs in real time:

```powershell
az ml job create --file pipelines/feature_pipeline.yml --stream
```

---

## Part IV – Results

### Validation Metrics

| Metric | Train | Validation |
|---|---|---|
| RMSE | — | — |
| MAE | — | — |
| R² | — | — |
| NASA Score | — | — |

---

### Top Features Selected by Genetic Algorithm

| Rank | Feature | Importance |
|---|---|---|
| 1 | — | — |
| 2 | — | — |
| 3 | — | — |
| 4 | — | — |
| 5 | — | — |

---

### Pipeline Runtime

| Step | Details | Runtime |
|---|---|---|
| Databricks ETL | Notebooks 05 & 06 | — |
| extract_features (train) | EfficientFCParameters, window=30 | — |
| extract_features (test) | EfficientFCParameters, window=30 | — |
| reduce_features | variance → correlation → MI | — |
| genetic_algorithm | 50 population, 30 generations | — |
| split_dataset | 80/20 stratified split | — |
| train_evaluate | RandomForest, 200 estimators | — |
| **Total Azure ML Pipeline** | | — |

---

## Summary

| Component | Purpose | Key Output |
|---|---|---|
| `extract_features` | Rolling window tsfresh extraction | Feature matrix per engine per cycle |
| `reduce_features` | Variance, correlation, MI filters | Reduced feature set |
| `genetic_algorithm` | DEAP binary GA optimisation | Optimal feature subset |
| `split_dataset` | 80/20 stratified train/validation split | Train and val Parquet |
| `train_evaluate` | RandomForest training and evaluation | RMSE —, R² — |
