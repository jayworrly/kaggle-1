import pandas as pd
import numpy as np
from scipy import stats
from ..utils.config import get_config
from ..utils.helpers import save_report

class DataCleaner:
    """Class to handle data cleaning operations."""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.df = None
        self.cleaned_df = None
        self.outlier_threshold = self.config['analysis']['outlier_threshold']
        
    def load_data(self, file_path=None):
        """Load the dataset for cleaning."""
        if file_path is None:
            file_path = self.config['dataset']['raw_file']
        
        self.df = pd.read_csv(file_path)
        self.cleaned_df = self.df.copy()
        print(f"Loaded dataset with {len(self.df)} rows for cleaning")
        return self.df
    
    def remove_outliers_iqr(self, columns=None):
        """Remove outliers using IQR method."""
        if self.cleaned_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        if columns is None:
            columns = self.cleaned_df.select_dtypes(include=[np.number]).columns
        
        original_count = len(self.cleaned_df)
        removed_count = 0
        
        print("Removing outliers using IQR method:")
        print("=" * 40)
        
        for column in columns:
            if column in self.cleaned_df.columns:
                Q1 = self.cleaned_df[column].quantile(0.25)
                Q3 = self.cleaned_df[column].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - self.outlier_threshold * IQR
                upper_bound = Q3 + self.outlier_threshold * IQR
                
                # Count outliers before removal
                outliers = self.cleaned_df[(self.cleaned_df[column] < lower_bound) | 
                                         (self.cleaned_df[column] > upper_bound)]
                outlier_count = len(outliers)
                
                # Remove outliers
                self.cleaned_df = self.cleaned_df[(self.cleaned_df[column] >= lower_bound) & 
                                                (self.cleaned_df[column] <= upper_bound)]
                
                removed_count += outlier_count
                print(f"{column}: Removed {outlier_count} outliers ({outlier_count/original_count*100:.1f}%)")
        
        final_count = len(self.cleaned_df)
        print(f"\nTotal rows removed: {removed_count} ({removed_count/original_count*100:.1f}%)")
        print(f"Remaining rows: {final_count}")
        
        return self.cleaned_df
    
    def remove_outliers_zscore(self, columns=None, threshold=3):
        """Remove outliers using Z-score method."""
        if self.cleaned_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        if columns is None:
            columns = self.cleaned_df.select_dtypes(include=[np.number]).columns
        
        original_count = len(self.cleaned_df)
        removed_count = 0
        
        print("Removing outliers using Z-score method:")
        print("=" * 40)
        
        for column in columns:
            if column in self.cleaned_df.columns:
                # Calculate z-scores
                z_scores = np.abs(stats.zscore(self.cleaned_df[column].dropna()))
                
                # Count outliers before removal
                outlier_mask = z_scores > threshold
                outlier_count = outlier_mask.sum()
                
                # Remove outliers
                self.cleaned_df = self.cleaned_df[z_scores <= threshold]
                
                removed_count += outlier_count
                print(f"{column}: Removed {outlier_count} outliers ({outlier_count/original_count*100:.1f}%)")
        
        final_count = len(self.cleaned_df)
        print(f"\nTotal rows removed: {removed_count} ({removed_count/original_count*100:.1f}%)")
        print(f"Remaining rows: {final_count}")
        
        return self.cleaned_df
    
    def handle_missing_values(self, strategy='drop'):
        """Handle missing values in the dataset."""
        if self.cleaned_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        original_count = len(self.cleaned_df)
        missing_before = self.cleaned_df.isnull().sum().sum()
        
        print(f"Handling missing values (strategy: {strategy}):")
        print("=" * 40)
        print(f"Missing values before: {missing_before}")
        
        if strategy == 'drop':
            self.cleaned_df = self.cleaned_df.dropna()
        elif strategy == 'fill_mean':
            # Fill numerical columns with mean
            numerical_cols = self.cleaned_df.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if self.cleaned_df[col].isnull().sum() > 0:
                    mean_val = self.cleaned_df[col].mean()
                    self.cleaned_df[col].fillna(mean_val, inplace=True)
        elif strategy == 'fill_median':
            # Fill numerical columns with median
            numerical_cols = self.cleaned_df.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if self.cleaned_df[col].isnull().sum() > 0:
                    median_val = self.cleaned_df[col].median()
                    self.cleaned_df[col].fillna(median_val, inplace=True)
        
        final_count = len(self.cleaned_df)
        missing_after = self.cleaned_df.isnull().sum().sum()
        removed_count = original_count - final_count
        
        print(f"Missing values after: {missing_after}")
        print(f"Rows removed: {removed_count} ({removed_count/original_count*100:.1f}%)")
        print(f"Remaining rows: {final_count}")
        
        return self.cleaned_df
    
    def remove_duplicates(self):
        """Remove duplicate rows."""
        if self.cleaned_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        original_count = len(self.cleaned_df)
        duplicate_count = self.cleaned_df.duplicated().sum()
        
        print(f"Removing duplicates:")
        print("=" * 40)
        print(f"Duplicate rows found: {duplicate_count}")
        
        self.cleaned_df = self.cleaned_df.drop_duplicates()
        
        final_count = len(self.cleaned_df)
        print(f"Rows removed: {duplicate_count} ({duplicate_count/original_count*100:.1f}%)")
        print(f"Remaining rows: {final_count}")
        
        return self.cleaned_df
    
    def clean_salary_data(self):
        """Clean salary-related data specifically."""
        if self.cleaned_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print("Cleaning salary data:")
        print("=" * 40)
        
        # Remove negative salaries
        negative_salaries = (self.cleaned_df['salary_usd'] < 0).sum()
        self.cleaned_df = self.cleaned_df[self.cleaned_df['salary_usd'] >= 0]
        print(f"Removed {negative_salaries} negative salaries")
        
        # Remove extremely high salaries (likely data errors)
        # Use 99th percentile as threshold
        salary_threshold = self.cleaned_df['salary_usd'].quantile(0.99)
        extreme_salaries = (self.cleaned_df['salary_usd'] > salary_threshold).sum()
        self.cleaned_df = self.cleaned_df[self.cleaned_df['salary_usd'] <= salary_threshold]
        print(f"Removed {extreme_salaries} extremely high salaries (>${salary_threshold:,.0f})")
        
        return self.cleaned_df
    
    def clean_experience_data(self):
        """Clean experience-related data."""
        if self.cleaned_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print("Cleaning experience data:")
        print("=" * 40)
        
        # Remove negative experience years
        negative_exp = (self.cleaned_df['years_experience'] < 0).sum()
        self.cleaned_df = self.cleaned_df[self.cleaned_df['years_experience'] >= 0]
        print(f"Removed {negative_exp} negative experience years")
        
        # Remove extremely high experience years (likely data errors)
        exp_threshold = self.cleaned_df['years_experience'].quantile(0.99)
        extreme_exp = (self.cleaned_df['years_experience'] > exp_threshold).sum()
        self.cleaned_df = self.cleaned_df[self.cleaned_df['years_experience'] <= exp_threshold]
        print(f"Removed {extreme_exp} extremely high experience years (>{exp_threshold:.0f} years)")
        
        return self.cleaned_df
    
    def apply_standard_cleaning(self):
        """Apply standard cleaning procedures."""
        if self.cleaned_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print("Applying standard cleaning procedures:")
        print("=" * 50)
        
        original_count = len(self.cleaned_df)
        
        # Step 1: Remove duplicates
        self.remove_duplicates()
        
        # Step 2: Handle missing values
        self.handle_missing_values(strategy='drop')
        
        # Step 3: Clean salary data
        self.clean_salary_data()
        
        # Step 4: Clean experience data
        self.clean_experience_data()
        
        # Step 5: Remove outliers from key numerical columns
        key_columns = ['salary_usd', 'remote_ratio', 'years_experience', 'job_description_length']
        self.remove_outliers_iqr(columns=key_columns)
        
        final_count = len(self.cleaned_df)
        removed_total = original_count - final_count
        
        print(f"\nCleaning Summary:")
        print(f"Original rows: {original_count}")
        print(f"Final rows: {final_count}")
        print(f"Total removed: {removed_total} ({removed_total/original_count*100:.1f}%)")
        
        return self.cleaned_df
    
    def save_cleaned_data(self, output_path=None):
        """Save the cleaned dataset."""
        if self.cleaned_df is None:
            raise ValueError("No cleaned data available. Run cleaning first.")
        
        if output_path is None:
            output_path = self.config['dataset']['cleaned_file']
        
        self.cleaned_df.to_csv(output_path, index=False)
        print(f"Cleaned dataset saved to: {output_path}")
        
        return output_path
    
    def generate_cleaning_report(self):
        """Generate a comprehensive cleaning report."""
        if self.cleaned_df is None:
            raise ValueError("No cleaned data available. Run cleaning first.")
        
        original_count = len(self.df)
        final_count = len(self.cleaned_df)
        removed_count = original_count - final_count
        
        report = []
        report.append("AI Job Market Dataset - Data Cleaning Report")
        report.append("=" * 60)
        report.append("")
        
        report.append("1. CLEANING SUMMARY")
        report.append("-" * 20)
        report.append(f"Original dataset size: {original_count} rows")
        report.append(f"Final dataset size: {final_count} rows")
        report.append(f"Rows removed: {removed_count} ({removed_count/original_count*100:.1f}%)")
        report.append("")
        
        report.append("2. CLEANING STEPS APPLIED")
        report.append("-" * 25)
        report.append("• Removed duplicate rows")
        report.append("• Removed rows with missing values")
        report.append("• Cleaned salary data (removed negative and extreme values)")
        report.append("• Cleaned experience data (removed negative and extreme values)")
        report.append("• Removed outliers using IQR method for key numerical columns")
        report.append("")
        
        report.append("3. DATA QUALITY IMPROVEMENTS")
        report.append("-" * 30)
        report.append("• Improved data consistency")
        report.append("• Removed potential data entry errors")
        report.append("• Enhanced model training reliability")
        report.append("• Reduced noise in the dataset")
        
        report_text = "\n".join(report)
        save_report(report_text, 'data_cleaning_report.txt')
        
        return report_text

def run_data_cleaning(input_file=None, output_file=None):
    """Convenience function to run complete data cleaning."""
    cleaner = DataCleaner()
    cleaner.load_data(input_file)
    cleaner.apply_standard_cleaning()
    cleaner.save_cleaned_data(output_file)
    report = cleaner.generate_cleaning_report()
    print("Data cleaning completed. Check outputs/reports/ for cleaning report.")
    return cleaner

if __name__ == "__main__":
    run_data_cleaning() 