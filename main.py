#!/usr/bin/env python3
"""
Main entry point for the AI Job Market Analysis project.
"""

import argparse
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.downloader import DatasetDownloader
from src.data.database import DatabaseManager
from src.utils.config import get_config

def main():
    parser = argparse.ArgumentParser(description="AI Job Market Analysis")
    parser.add_argument("--download", action="store_true", help="Download the dataset")
    parser.add_argument("--create-db", action="store_true", help="Create the database")
    parser.add_argument("--import-db", action="store_true", help="Import data to database")
    parser.add_argument("--full-setup", action="store_true", help="Run full setup (download + create db + import)")
    
    args = parser.parse_args()
    
    if not any([args.download, args.create_db, args.import_db, args.full_setup]):
        parser.print_help()
        return
    
    try:
        config = get_config()
        
        if args.download or args.full_setup:
            print("Downloading dataset...")
            downloader = DatasetDownloader(config)
            if not downloader.download():
                print("Failed to download dataset")
                return
        
        if args.create_db or args.full_setup:
            print("Creating database...")
            db_manager = DatabaseManager(config)
            if not db_manager.create_database():
                print("Failed to create database")
                return
        
        if args.import_db or args.full_setup:
            print("Importing data to database...")
            db_manager = DatabaseManager(config)
            if not db_manager.import_csv_to_postgres():
                print("Failed to import data to database")
                return
        
        print("Operation completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 