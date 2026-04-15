# Lab 3 - Data Preprocessing on Azure

## Overview
This lab implements a data preprocessing pipeline using Azure Databricks to process Amazon Electronics review data through a medallion architecture (Bronze → Silver → Gold layers).

This pipeline follows the Medallion (Lakehouse) architecture:
- **Bronze layer**: raw JSON data as originally ingested  
- **Silver layer**: cleaned and validated data  
- **Gold layer**: curated dataset ready for analytics and machine learning  

This layered approach improves data quality step-by-step and ensures that only reliable data reaches the final stage.

---

## Technologies Used

- **Azure Databricks** – used to run and manage the data processing pipeline  
- **Apache Spark (PySpark)** – used for large-scale data transformation  
- **Azure Data Lake Storage Gen2** – stores data across Bronze, Silver, and Gold layers  
- **Parquet format** – efficient column-based storage for faster querying  
- **Databricks Jobs** – used to automate and schedule the pipeline  
- **Python (PySpark)** – used to implement ETL logic  

---

## Pipeline Implementation

In this lab, I built a data pipeline in Azure Databricks to clean and organize Amazon review data:

### 1. Databricks Notebooks
Created 3 Databricks notebooks that work together:
- **Notebook 1**: Loaded review data and cleaned it (removed missing values, fixed invalid ratings)  
- **Notebook 2**: Added product information such as brand, title, and price  
- **Notebook 3**: Saved the final clean dataset to the Gold layer  

---

### 2. Job Orchestration

Set up a Databricks Job to run all three notebooks automatically in sequence:
- Created a job called `lab3_data_preprocessing_job`  
- Added notebooks as dependent tasks (each runs after the previous one)  
- Successfully executed the pipeline in ~9 minutes  

<img width="751" height="608" alt="lab3" src="https://github.com/user-attachments/assets/a48e03ff-d0af-4b64-9b93-6eacf9057f9e" />

---

### 3. Scheduling

- Configured the job to run daily  
- Enables fully automated pipeline execution  

This is important in real-world systems where data is continuously generated, as it ensures the dataset stays updated without manual intervention.

<img width="966" height="548" alt="schedule" src="https://github.com/user-attachments/assets/00797c2a-f882-4566-b162-c6d4f50d9f08" />

---

## Final Output

The end result is a clean, enriched dataset stored in the curated (Gold) layer, ready for analysis and machine learning.

Pipeline notebooks:
- `01_load_and_clean_reviews`  
- `02_enrich_with_metadata`  
- `03_write_gold_features_v1`  

---

## ETL Mapping

### Extract (Notebook 1)
- Data is extracted from the processed container (Silver layer from Lab 2)

### Transform (Notebooks 1 & 2)
- **Notebook 1**:
  - Removes missing values  
  - Filters invalid ratings  
  - Cleans review text  

- **Notebook 2**:
  - Joins review data with product metadata (brand, title, price)  

This step is important because cleaning early prevents errors from propagating through the pipeline, and enrichment adds useful context that improves downstream analysis and machine learning performance.

### Load (Notebook 3)
- Final dataset is written to the curated (Gold) layer  

The data is stored in **Parquet format**, which is more efficient than raw formats like JSON because it reduces storage size and improves query performance.

**Summary:**  
Extract → Transform → Load  
(Grab data → clean & enrich → save final dataset)

---

## Additional Enrichment Ideas

### Price & Category Features
- Price brackets:
  - Budget (< $50)  
  - Mid-range ($50–200)  
  - Premium (> $200)  
- Rating categories:
  - Poor (1–2 stars)  
  - Average (3 stars)  
  - Good (4–5 stars)  

### Reviewer Behaviour Features
- Number of reviews per user  
- Rating consistency patterns  

### Text-Based Features
- Word count / character count  
- Sentiment or emotion classification  

---

## Visualizations

### 1. Rating Distribution
A bar chart showing the number of reviews for each rating (1–5).

<img width="746" height="449" alt="rating distribution" src="https://github.com/user-attachments/assets/fe5b08a6-b5c1-489c-b003-b97f123a74cb" />

- Almost 60% of reviews are 5-star  
- Strong bias toward extreme ratings (especially positive)  
- Only ~6.5% are 1-star reviews  
- Low percentage of 1–2 star reviews suggests generally good product quality  
- This imbalance could introduce bias in machine learning models, as models may become more likely to predict positive outcomes  

---

### 2. Review Length vs Rating
Box plots comparing review length across different ratings.

<img width="743" height="445" alt="review length" src="https://github.com/user-attachments/assets/460cf8db-1e83-4da9-be35-01a824351684" />

- Median review length is similar across ratings  
- 1-star reviews show higher variability in length  
- Positive reviews range from very short to detailed  
- Users with strong opinions (very positive or negative) tend to write longer reviews  
- This can be useful for sentiment analysis, as extreme opinions often provide richer textual data  

---

## Final Reflection

This lab helped me understand how raw data is transformed into a structured and reliable dataset using cloud-based tools. It also demonstrated how data pipelines can be automated using Databricks Jobs, making them suitable for real-world applications where data is continuously updated.
