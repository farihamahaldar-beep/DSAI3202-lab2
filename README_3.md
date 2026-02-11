Overview: This lab implements a data preprocessing pipeline using Azure Databricks to process Amazon Electronics review data through a medallion architecture (Bronze → Silver → Gold layers).

In this lab, I built a data pipeline in Azure Databricks to clean and organize Amazon review data:

1. Created 3 Databricks notebooks that work together:
   - First notebook: Loaded review data and cleaned it (removed missing values, fixed bad ratings)
   - Second notebook: Added product information like brand, title, and price to the reviews
   - Third notebook: Saved the final clean dataset to the "gold" layer

2. Set up a Databricks Job to run all three notebooks automatically in order:
   - Created a job called `lab3_data_preprocessing_job`
   - Added all three notebooks as tasks with dependencies (each one waits for the previous to finish)
   - Ran the job successfully - all tasks completed in about 9 minutes

    <img width="751" height="608" alt="lab3" src="https://github.com/user-attachments/assets/a48e03ff-d0af-4b64-9b93-6eacf9057f9e" />


3. Scheduled the job to run automatically:
   - Set it up to run daily so the data stays updated
   - This means the pipeline can run on its own without me clicking anything

   <img width="966" height="548" alt="Screenshot 2026-02-10 225629" src="https://github.com/user-attachments/assets/00797c2a-f882-4566-b162-c6d4f50d9f08" />


The end result is a clean, enriched dataset in the curated container that's ready for analysis and machine learning.

Pipeline consists of three sequential notebooks: 01_load_and_clean_reviews, 02_enrich_with_metadata, 03_write_gold_features_v1



How do the notebooks map to the ETL process?

Extract (Notebook 1):
- We extract/pull the review data from the processed container where it was stored in Lab 2

Transform (Notebooks 1 & 2):
- Notebook 1: We clean the data by removing bad reviews (missing info, invalid ratings, reviews that are too short)
- Notebook 2: We add more information by joining the reviews with product details like brand, title, and price
- Basically we're fixing and improving the data to make it useful

Load (Notebook 3):
- We save the final clean and enriched data to the curated container
- This is our finished product that's ready to use for analysis or machine learning

So it's: grabing the data → clean it up and add stuff → save the final version
