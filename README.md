# Lab 2: Data Ingestion Pipeline on Azure
1. Start by setting up the data lake (create storage acc, organize raw, processed and curated)
   <img width="997" height="259" alt="image" src="https://github.com/user-attachments/assets/c0549dde-6a63-4fa9-ac93-0317d1c94eca" />

2. Created a VM, downloaded the data Amazon Electronics 
<img width="975" height="260" alt="image" src="https://github.com/user-attachments/assets/81cea223-2259-4b05-9a94-84e4609db95d" />

3. Created an SSHA Key and 
4. Used a tool called azcopy to push the fixed file (meta_Electronics_fixed.json) into my cloud storage using terminal
<img width="975" height="299" alt="image" src="https://github.com/user-attachments/assets/5a815dd4-e260-451e-8874-b8d0ad9ab35a" />

<img width="975" height="459" alt="image" src="https://github.com/user-attachments/assets/3c9a59fd-7e98-4dc9-89ef-4f9b627db59e" />
<img width="975" height="303" alt="image" src="https://github.com/user-attachments/assets/3b28cb87-ab58-46c5-974a-75605067c563" />
<img width="975" height="530" alt="image" src="https://github.com/user-attachments/assets/b333e762-5091-4604-bbc2-7550f4a318ce" />

 5. I used Azure Data Factory to build a pipeline that automatically moves and changes the data.
<img width="975" height="480" alt="image" src="https://github.com/user-attachments/assets/31b0aa47-8e25-4501-826c-7ad5b23ba70c" />
<img width="975" height="344" alt="image" src="https://github.com/user-attachments/assets/d30fe0ee-f7f7-4fda-bf40-8b7a6587bc9f" />

6. The pipeline worked perfectly. When I checked my processed/reviews folder, Azure had automatically created a separate folder for every year (from 1999 to 2014).
<img width="975" height="493" alt="image" src="https://github.com/user-attachments/assets/565ee909-9e5b-447e-92a7-ef0cd4731cdb" />
