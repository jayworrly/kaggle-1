import kaggle
import os

# Set the dataset name and path
dataset_name = 'bismasajjad/global-ai-job-market-and-salary-trends-2025'

# Create a directory for the dataset if it doesn't exist
data_dir = 'data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Download the dataset
print(f"Downloading dataset: {dataset_name}")
kaggle.api.dataset_download_files(dataset_name, path=data_dir, unzip=True)
print("Dataset downloaded successfully.") 