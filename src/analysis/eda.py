import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from ..utils.config import get_config
from ..utils.helpers import save_plot, save_report

class EDAAnalyzer:
    """Class to handle Exploratory Data Analysis."""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.df = None
        
    def load_data(self, file_path=None):
        """Load the dataset for analysis."""
        if file_path is None:
            file_path = self.config['dataset']['raw_file']
        
        self.df = pd.read_csv(file_path)
        print(f"Loaded dataset with {len(self.df)} rows and {len(self.df.columns)} columns")
        return self.df
    
    def basic_info(self):
        """Display basic information about the dataset."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print("Dataset Information:")
        self.df.info()
        
        print("\nFirst 5 rows of the dataset:")
        print(self.df.head())
        
        print("\nDescriptive Statistics:")
        print(self.df.describe())
        
        return {
            'shape': self.df.shape,
            'info': self.df.info(),
            'head': self.df.head(),
            'describe': self.df.describe()
        }
    
    def check_missing_values(self):
        """Check for missing values in the dataset."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        missing_values = self.df.isnull().sum()
        missing_percentage = (missing_values / len(self.df)) * 100
        
        missing_df = pd.DataFrame({
            'Column': missing_values.index,
            'Missing_Count': missing_values.values,
            'Missing_Percentage': missing_percentage.values
        }).sort_values('Missing_Percentage', ascending=False)
        
        print("Missing Values Analysis:")
        print(missing_df)
        
        return missing_df
    
    def check_duplicates(self):
        """Check for duplicate rows."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        duplicate_count = self.df.duplicated().sum()
        print(f"Duplicate Rows: {duplicate_count}")
        
        return duplicate_count
    
    def analyze_categorical_columns(self):
        """Analyze categorical columns."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        categorical_columns = ['experience_level', 'employment_type', 'company_size', 'education_required']
        
        print("Unique values in key categorical columns:")
        for col in categorical_columns:
            if col in self.df.columns:
                unique_values = self.df[col].unique()
                print(f"{col}: {unique_values}")
        
        return {col: self.df[col].unique() for col in categorical_columns if col in self.df.columns}
    
    def create_summary_plots(self):
        """Create summary plots for key variables."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('AI Job Market Dataset - Summary Plots', fontsize=16)
        
        # Salary distribution
        axes[0, 0].hist(self.df['salary_usd'].dropna(), bins=30, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Salary Distribution (USD)')
        axes[0, 0].set_xlabel('Salary (USD)')
        axes[0, 0].set_ylabel('Frequency')
        
        # Experience level distribution
        experience_counts = self.df['experience_level'].value_counts()
        axes[0, 1].pie(experience_counts.values, labels=experience_counts.index, autopct='%1.1f%%')
        axes[0, 1].set_title('Experience Level Distribution')
        
        # Employment type distribution
        employment_counts = self.df['employment_type'].value_counts()
        axes[1, 0].bar(employment_counts.index, employment_counts.values, color='lightcoral')
        axes[1, 0].set_title('Employment Type Distribution')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Company size distribution
        company_size_counts = self.df['company_size'].value_counts()
        axes[1, 1].bar(company_size_counts.index, company_size_counts.values, color='lightgreen')
        axes[1, 1].set_title('Company Size Distribution')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        save_plot(fig, 'eda_summary_plots.png')
        plt.close()
        
        return fig
    
    def generate_report(self):
        """Generate a comprehensive EDA report."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        report = []
        report.append("AI Job Market Dataset - Exploratory Data Analysis Report")
        report.append("=" * 60)
        report.append("")
        
        # Basic information
        report.append("1. DATASET OVERVIEW")
        report.append("-" * 20)
        report.append(f"Total rows: {len(self.df)}")
        report.append(f"Total columns: {len(self.df.columns)}")
        report.append(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        report.append("")
        
        # Missing values
        missing_df = self.check_missing_values()
        report.append("2. MISSING VALUES")
        report.append("-" * 20)
        for _, row in missing_df.iterrows():
            if row['Missing_Count'] > 0:
                report.append(f"{row['Column']}: {row['Missing_Count']} ({row['Missing_Percentage']:.1f}%)")
        if missing_df['Missing_Count'].sum() == 0:
            report.append("No missing values found in the dataset.")
        report.append("")
        
        # Duplicates
        duplicate_count = self.check_duplicates()
        report.append("3. DUPLICATE ROWS")
        report.append("-" * 20)
        report.append(f"Duplicate rows: {duplicate_count}")
        report.append("")
        
        # Categorical analysis
        report.append("4. CATEGORICAL VARIABLES")
        report.append("-" * 20)
        categorical_data = self.analyze_categorical_columns()
        for col, values in categorical_data.items():
            report.append(f"{col}: {len(values)} unique values")
        report.append("")
        
        # Numerical summary
        report.append("5. NUMERICAL VARIABLES SUMMARY")
        report.append("-" * 20)
        numerical_cols = self.df.select_dtypes(include=['number']).columns
        for col in numerical_cols:
            if col in self.df.columns:
                report.append(f"{col}:")
                report.append(f"  Mean: {self.df[col].mean():.2f}")
                report.append(f"  Median: {self.df[col].median():.2f}")
                report.append(f"  Std: {self.df[col].std():.2f}")
                report.append(f"  Min: {self.df[col].min()}")
                report.append(f"  Max: {self.df[col].max()}")
                report.append("")
        
        report_text = "\n".join(report)
        save_report(report_text, 'eda_report.txt')
        
        return report_text

def run_eda_analysis(file_path=None):
    """Convenience function to run complete EDA analysis."""
    analyzer = EDAAnalyzer()
    analyzer.load_data(file_path)
    analyzer.basic_info()
    analyzer.create_summary_plots()
    report = analyzer.generate_report()
    print("EDA analysis completed. Check outputs/plots/ and outputs/reports/ for results.")
    return analyzer

if __name__ == "__main__":
    run_eda_analysis() 