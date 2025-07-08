#!/usr/bin/env python3
"""
Full pipeline script to run the complete AI Job Market Analysis.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.downloader import DatasetDownloader
from src.data.database import DatabaseManager
from src.utils.config import get_config

def run_full_pipeline():
    """Run the complete analysis pipeline."""
    print("Starting AI Job Market Analysis Pipeline...")
    
    try:
        config = get_config()
        
        # Step 1: Download dataset
        print("\n1. Downloading dataset...")
        downloader = DatasetDownloader(config)
        if not downloader.download():
            print("Failed to download dataset. Exiting.")
            return False
        
        # Step 2: Create database
        print("\n2. Creating database...")
        db_manager = DatabaseManager(config)
        if not db_manager.create_database():
            print("Failed to create database. Exiting.")
            return False
        
        # Step 3: Import data to database
        print("\n3. Importing data to database...")
        csv_file = config['dataset']['raw_file']
        if not db_manager.import_csv_to_postgres(csv_file):
            print("Failed to import data to database. Exiting.")
            return False
        
        print("\nPipeline completed successfully!")
        print("\nNext steps:")
        print("- Run EDA analysis: python scripts/run_analysis.py")
        print("- Run modeling: python scripts/run_modeling.py")
        
        return True
        
    except Exception as e:
        print(f"Error in pipeline: {e}")
        return False

if __name__ == "__main__":
    success = run_full_pipeline()
    sys.exit(0 if success else 1) 