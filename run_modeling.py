#!/usr/bin/env python3
"""
AI Job Market Analysis - Modeling Pipeline Launcher

This script runs the complete modeling pipeline including:
- Regression modeling for salary prediction
- Classification modeling for job categorization
- Clustering analysis for market segmentation

Usage:
    python run_modeling.py [--regression] [--classification] [--clustering] [--all]
"""

import sys
import os
import argparse
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

# Now we can import our modules
from src.models.regression import run_salary_prediction
from src.models.classification import run_job_classification
from src.models.clustering import run_job_clustering
from src.utils.config import get_config

def main():
    """Run the modeling pipeline based on command line arguments."""
    parser = argparse.ArgumentParser(description='AI Job Market Modeling Pipeline')
    parser.add_argument('--regression', action='store_true', 
                       help='Run regression modeling for salary prediction')
    parser.add_argument('--classification', action='store_true', 
                       help='Run classification modeling for job categorization')
    parser.add_argument('--clustering', action='store_true', 
                       help='Run clustering analysis for market segmentation')
    parser.add_argument('--all', action='store_true', 
                       help='Run all modeling tasks')
    parser.add_argument('--target-column', type=str, default='experience_level',
                       help='Target column for classification (default: experience_level)')
    parser.add_argument('--n-clusters', type=int, default=None,
                       help='Number of clusters for clustering (default: auto-detect)')
    
    args = parser.parse_args()
    
    # If no specific task is selected, run all
    if not any([args.regression, args.classification, args.clustering, args.all]):
        args.all = True
    
    config = get_config()
    
    print("=" * 60)
    print("AI JOB MARKET ANALYSIS - MODELING PIPELINE")
    print("=" * 60)
    
    try:
        if args.all or args.regression:
            print("\n" + "="*40)
            print("RUNNING REGRESSION MODELING")
            print("="*40)
            run_salary_prediction()
        
        if args.all or args.classification:
            print("\n" + "="*40)
            print("RUNNING CLASSIFICATION MODELING")
            print("="*40)
            run_job_classification(target_column=args.target_column)
        
        if args.all or args.clustering:
            print("\n" + "="*40)
            print("RUNNING CLUSTERING ANALYSIS")
            print("="*40)
            run_job_clustering(n_clusters=args.n_clusters)
        
        print("\n" + "="*60)
        print("MODELING PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nGenerated outputs:")
        print("• Plots: outputs/plots/")
        print("• Models: outputs/models/")
        print("• Reports: outputs/reports/")
        
    except Exception as e:
        print(f"\nERROR: Modeling pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 