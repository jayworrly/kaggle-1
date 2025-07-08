import kaggle
import os
from pathlib import Path
from ..utils.config import get_config

class DatasetDownloader:
    """Class to handle dataset downloading from Kaggle."""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.dataset_name = self.config['dataset']['name']
        self.data_dir = Path('data')
    
    def download(self):
        """Download the dataset from Kaggle."""
        # Create data directory if it doesn't exist
        self.data_dir.mkdir(exist_ok=True)
        
        print(f"Downloading dataset: {self.dataset_name}")
        try:
            kaggle.api.dataset_download_files(
                self.dataset_name, 
                path=str(self.data_dir), 
                unzip=True
            )
            print("Dataset downloaded successfully.")
            return True
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return False
    
    def check_dataset_exists(self):
        """Check if the dataset file exists."""
        raw_file = Path(self.config['dataset']['raw_file'])
        return raw_file.exists()

def download_dataset():
    """Convenience function to download the dataset."""
    downloader = DatasetDownloader()
    return downloader.download()

if __name__ == "__main__":
    download_dataset() 