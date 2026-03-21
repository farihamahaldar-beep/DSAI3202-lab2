# Lab 5 – Scalable Feature Extraction and Selection for Predictive Maintenance

> **Note:** The commit history on this repository starts two weeks after the lab deadline.
> This is not when the work was started — development began on time, but I inadvertently
> created a new repository instead of branching from the correct one. The mistake was only
> noticed later.

---

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
└── .gitignore
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

`MinimalFCParameters` was used to keep extraction fast on the Standard compute instance while still producing meaningful statistical features per sensor per window.

| Parameter | Value |
|---|---|
| Feature set | `MinimalFCParameters` |
| Window size | 30 cycles |
| Output rows (train) | 14,184 |
| Output rows (val) | 3,547 |

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

The GA is preferred over purely filter-based methods because it can discover **feature combinations** that work well together, not just features that are individually strong. The GA reduced the input feature set down to just **8 features** — a 92%+ reduction — while achieving R² of 0.97.

| Parameter | Value |
|---|---|
| Population size | 50 |
| Generations | 30 |
| Crossover probability | 0.7 |
| Mutation probability | 0.2 |
| Selection | Tournament (size 3) |
| Fitness | 3-fold CV RMSE − α × feature ratio |
| Features in → out | 100 → 8 |

---

#### Split Dataset (`split_dataset`)

Splits the GA-selected training features into 80% training and 20% validation with a fixed random seed. Stratification bins RUL values to ensure equal distribution of engine health states across both splits. The split occurs after all feature selection steps to ensure no validation data influenced which features were selected or how they were extracted.

| Split | Rows |
|---|---|
| Train | 14,184 |
| Validation | 3,547 |

---

#### Train and Evaluate (`train_evaluate`)

Trains a **RandomForest Regressor** on the training split and evaluates on the validation split. The model is saved via joblib, predictions are saved as CSV, and all metrics are written to `metrics.json`.

| Parameter | Value |
|---|---|
| Model | RandomForest Regressor |
| n_estimators | 200 |
| max_depth | unlimited |
| OOB score | enabled |
| Training time | 1.7s |

Evaluation includes RMSE, MAE, R², the NASA asymmetric scoring function (which penalises late predictions more heavily than early ones), and feature importances.

---

### Reproducing the Experiment

#### Prerequisites

The following must be installed and configured before running the pipeline:

- [Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli) (v2.40+)
- Azure ML CLI extension: `az extension add -n ml`
- Python 3.9+
- Access to an Azure ML workspace with a registered compute cluster

Verify your setup:
```powershell
az --version
az ml --version
```

---

#### Step 1 — Clone the Repository

```powershell
git clone <your-repo-url>
cd LAB5
```

---

#### Step 2 — Log in to Azure

```powershell
az login
az account set --subscription <your-subscription-id>
```

---

#### Step 3 — Configure the Workspace

Set your workspace and resource group so you don't have to pass them on every command:

```powershell
az configure --defaults group=<your-resource-group> workspace=<your-workspace-name>
```

---

#### Step 4 — Verify the Datastore

The pipeline reads from the `blobkey` datastore which points to the `curated` container in Azure Data Lake. Confirm it exists:

```powershell
az ml datastore show --name blobkey --query "{account:account_name, container:container_name}"
```

Expected output:
```json
{
  "account": "datalake60306249",
  "container": "curated"
}
```

The following paths must exist in that container before running:
- `FD001/train_tsfresh_ready/`
- `FD001/test_tsfresh_ready/`
- `FD001/train_rul_labels/`

These are produced by running Databricks notebooks `05_load_and_preprocess_turbofan` and `06_feature_extraction_tsfresh` in order.

---

#### Step 5 — Register All Components

Run all five component registrations from the `LAB5/` root:

```powershell
az ml component create --file components/extract_features/component.yml
az ml component create --file components/reduce_features/component.yml
az ml component create --file components/genetic_algorithm/component.yml
az ml component create --file components/split_dataset/component.yml
az ml component create --file components/train_evaluate/component.yml
```

Verify they are registered:
```powershell
az ml component list --query "[].{name:name, version:version}" -o table
```

---

#### Step 6 — Submit the Pipeline

```powershell
az ml job create --file pipelines/feature_pipeline.yml
```

Stream logs in real time:
```powershell
az ml job stream --name <job-name>
```

Get the job name if you didn't note it:
```powershell
az ml job list --query "[0].name" -o tsv
```

---

#### Step 7 — Retrieve Outputs

Once the pipeline completes, download the metrics and predictions from the job outputs in Azure ML Studio:

1. Go to **Azure ML Studio** → **Jobs**
2. Click the completed job
3. Click **Outputs + logs**
4. Under `final_metrics/` download:
   - `metrics.json` — RMSE, MAE, R², NASA score
   - `val_predictions.csv` — per-row predictions vs ground truth
   - `feature_importances.csv` — ranked feature importances
5. Under `final_model/` download:
   - `random_forest_rul.joblib` — trained model

---

#### Modifying Hyperparameters

All pipeline hyperparameters are controlled from a single place — the `inputs` section at the top of `pipelines/feature_pipeline.yml`. No code changes are needed. Key parameters to experiment with:

| Parameter | Default | Effect |
|---|---|---|
| `window_size` | 30 | Larger = more context per window, slower extraction |
| `mi_top_k` | 100 | More features passed to GA, slower GA |
| `population_size` | 50 | Larger = better GA search, slower |
| `n_generations` | 30 | More generations = better convergence, slower |
| `n_estimators` | 200 | More trees = better accuracy, slower training |

---

## Part IV – Results

### Validation Metrics

| Metric | Train | Validation |
|---|---|---|
| RMSE | 3.54 | **7.05** |
| MAE | 2.21 | **4.39** |
| R² | 0.9928 | **0.9716** |
| OOB R² | — | **0.9738** |
| NASA Score | — | **4201.10** |

A validation R² of **0.97** means the model explains 97% of the variance in RUL — well above the typical 0.85–0.92 range reported for Random Forest on C-MAPSS FD001. The model achieves this with only **8 features** selected by the genetic algorithm out of hundreds extracted by tsfresh, demonstrating that the feature selection pipeline is highly effective.

The train/validation gap is small (R² 0.9928 vs 0.9716), indicating the model generalises well without significant overfitting.

The full Azure ML pipeline completed in **4 minutes 59 seconds** end-to-end.

---

### Top Features Selected by Genetic Algorithm

| Rank | Feature | Importance |
|---|---|---|
| 1 | `sensor_2_scaled__sum_values` | 27.3% |
| 2 | `sensor_15_scaled__maximum` | 18.7% |
| 3 | `sensor_20_scaled__minimum` | 14.4% |
| 4 | `sensor_9_scaled__sum_values` | 14.3% |
| 5 | `sensor_2_scaled__minimum` | 12.1% |
| 6 | `sensor_8_scaled__sum_values` | 8.3% |
| 7 | `op_setting_2_scaled__sum_values` | 2.6% |
| 8 | `op_setting_1_scaled__sum_values` | 2.3% |

`sensor_2` (total temperature at LPC outlet) dominates at 27.3% importance, consistent with it being one of the most established degradation indicators in the C-MAPSS FD001 literature. The GA selected only 8 features yet achieved R² of 0.97, confirming the pipeline selected physically meaningful signals rather than noise.

---

## Summary

| Component | Purpose | Key Output |
|---|---|---|
| `extract_features` | Rolling window tsfresh extraction | 17,731 rows × ~100 features |
| `reduce_features` | Variance, correlation, MI filters | Reduced feature set |
| `genetic_algorithm` | DEAP binary GA optimisation | 8 optimal features |
| `split_dataset` | 80/20 stratified train/validation split | 14,184 train / 3,547 val |
| `train_evaluate` | RandomForest training and evaluation | **RMSE 7.05, R² 0.97, 8 features** |
| **Total Pipeline** | End-to-end Azure ML run | **4m 59s** |
