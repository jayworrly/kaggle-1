#!/usr/bin/env python3
"""
Analysis pipeline script to run EDA and data analysis.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import get_config
from src.utils.helpers import ensure_directory
from src.analysis.eda import EDAAnalyzer
from src.analysis.outliers import OutlierAnalyzer
from src.analysis.cleaning import DataCleaner

def run_analysis_pipeline():
    """Run the complete analysis pipeline."""
    print("Starting AI Job Market Analysis Pipeline...")
    
    try:
        config = get_config()
        
        # Ensure output directories exist
        ensure_directory(config['outputs']['plots'])
        ensure_directory(config['outputs']['reports'])
        
        print("\n1. Running Exploratory Data Analysis...")
        eda_analyzer = EDAAnalyzer(config)
        eda_analyzer.load_data()
        eda_analyzer.basic_info()
        eda_analyzer.create_summary_plots()
        eda_analyzer.generate_report()
        print("   ✅ EDA analysis completed")
        
        print("\n2. Running Outlier Analysis...")
        outlier_analyzer = OutlierAnalyzer(config)
        outlier_analyzer.load_data()
        outlier_analyzer.analyze_numerical_outliers()
        outlier_analyzer.create_outlier_plots()
        outlier_analyzer.create_salary_outlier_analysis()
        outlier_analyzer.generate_outlier_report()
        print("   ✅ Outlier analysis completed")
        
        print("\n3. Running Data Cleaning...")
        data_cleaner = DataCleaner(config)
        data_cleaner.load_data()
        data_cleaner.apply_standard_cleaning()
        data_cleaner.save_cleaned_data()
        data_cleaner.generate_cleaning_report()
        print("   ✅ Data cleaning completed")
        
        print("\nAnalysis pipeline completed successfully!")
        print("\nGenerated outputs:")
        print("- Plots: outputs/plots/")
        print("- Reports: outputs/reports/")
        print("- Cleaned data: data/ai_job_dataset_cleaned.csv")
        print("\nNext steps:")
        print("- Run modeling: python scripts/run_modeling.py")
        
        return True
        
    except Exception as e:
        print(f"Error in analysis pipeline: {e}")
        return False

if __name__ == "__main__":
    success = run_analysis_pipeline()
    sys.exit(0 if success else 1) 