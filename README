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



1. How do the notebooks map to the ETL process?

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

2. What other enrichment can you do to the current Gold layer?
   - Price and Category Features:
      - Price brackets: "budget" (under $50), "mid-range" ($50-200), "premium" (over $200)
      - Rating categories: "poor" (1-2 stars), "average" (3 stars), "good" (4-5 stars)
   - Reviewer Behavior Features:
      - Number of reviews each person has written (are they active reviewers or one-time users?)
      - Review consistency (do they always give same ratings or vary?)
   - Text Analysis Features:
      - Word count and character count of review text
      - Emotion analysis (positive/negative/neutral emotion detection)

4. VISUALIZATIONS
      1. Rating distribution
         A bar chart showing how many reviews have each star rating (1-5)
         <img width="746" height="449" alt="image" src="https://github.com/user-attachments/assets/fe5b08a6-b5c1-489c-b003-b97f123a74cb" />
         - Almost 60% of all reviews are 5-star ratings.
         - The data shows a strong bias toward extreme ratings - many 5-stars (positive) and relatively fewer middle ratings.
         - Only 6.5% of reviews are 1-star indicating most customers are satisfied.
           This indicates that amazon products generally satisfy customers. The low percentage of 1-2 star reviews (10.4% combined) suggests decent product quality overall.

      3. Review Length vs Rating
         Box plots showing how long reviews are for each star rating.
         <img width="743" height="445" alt="image" src="https://github.com/user-attachments/assets/460cf8db-1e83-4da9-be35-01a824351684" />
         - The middle line in each box is around the same height i.e., all ratings have similar median lengths.
         - The 1-star box appears slightly taller, suggesting unhappy customers write reviews of varying lengths.
         - Happy customers review range from brief "Great product!" to detailed explanations.
           This indicates People who feel strongly (1-star or 5-star) sometimes write very long, detailed reviews. Those extremely long reviews (the dots way up high) are people who really wanted to share                 their experience in detail.


   
   
   
