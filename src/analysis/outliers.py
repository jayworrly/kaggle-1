import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from ..utils.config import get_config
from ..utils.helpers import save_plot, save_report

class OutlierAnalyzer:
    """Class to handle outlier detection and visualization."""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.df = None
        self.outlier_threshold = self.config['analysis']['outlier_threshold']
        
    def load_data(self, file_path=None):
        """Load the dataset for analysis."""
        if file_path is None:
            file_path = self.config['dataset']['raw_file']
        
        self.df = pd.read_csv(file_path)
        print(f"Loaded dataset with {len(self.df)} rows for outlier analysis")
        return self.df
    
    def detect_outliers_iqr(self, column):
        """Detect outliers using IQR method."""
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = self.df[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)]
        
        return outliers, lower_bound, upper_bound, Q1, Q3, IQR
    
    def detect_outliers_zscore(self, column, threshold=3):
        """Detect outliers using Z-score method."""
        z_scores = np.abs(stats.zscore(self.df[column].dropna()))
        outliers = self.df[z_scores > threshold]
        
        return outliers, z_scores
    
    def analyze_numerical_outliers(self):
        """Analyze outliers in numerical columns."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        numerical_columns = self.df.select_dtypes(include=[np.number]).columns
        outlier_summary = {}
        
        print("Outlier Analysis for Numerical Columns:")
        print("=" * 50)
        
        for column in numerical_columns:
            if column in self.df.columns:
                print(f"\n{column}:")
                
                # IQR method
                outliers_iqr, lower_bound, upper_bound, Q1, Q3, IQR = self.detect_outliers_iqr(column)
                
                # Z-score method
                outliers_zscore, z_scores = self.detect_outliers_zscore(column)
                
                outlier_summary[column] = {
                    'iqr_outliers': len(outliers_iqr),
                    'zscore_outliers': len(outliers_zscore),
                    'iqr_bounds': (lower_bound, upper_bound),
                    'quartiles': (Q1, Q3, IQR),
                    'mean': self.df[column].mean(),
                    'std': self.df[column].std()
                }
                
                print(f"  IQR outliers: {len(outliers_iqr)} ({len(outliers_iqr)/len(self.df)*100:.1f}%)")
                print(f"  Z-score outliers: {len(outliers_zscore)} ({len(outliers_zscore)/len(self.df)*100:.1f}%)")
                print(f"  IQR bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
                print(f"  Mean: {self.df[column].mean():.2f}")
                print(f"  Std: {self.df[column].std():.2f}")
        
        return outlier_summary
    
    def create_outlier_plots(self):
        """Create comprehensive outlier visualization plots."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        numerical_columns = self.df.select_dtypes(include=[np.number]).columns
        
        # Create subplots
        n_cols = min(3, len(numerical_columns))
        n_rows = (len(numerical_columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('Outlier Analysis - Box Plots and Histograms', fontsize=16)
        
        for idx, column in enumerate(numerical_columns):
            row = idx // n_cols
            col = idx % n_cols
            
            # Create subplot
            ax = axes[row, col]
            
            # Box plot
            self.df.boxplot(column=column, ax=ax)
            ax.set_title(f'{column} - Box Plot')
            ax.tick_params(axis='x', rotation=45)
            
            # Add outlier count annotation
            outliers_iqr, _, _, _, _, _ = self.detect_outliers_iqr(column)
            outlier_count = len(outliers_iqr)
            ax.text(0.02, 0.98, f'Outliers: {outlier_count}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Hide empty subplots
        for idx in range(len(numerical_columns), n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        save_plot(fig, 'outlier_boxplots.png')
        plt.close()
        
        # Create histogram plots with outlier highlighting
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('Outlier Analysis - Histograms with Outlier Highlighting', fontsize=16)
        
        for idx, column in enumerate(numerical_columns):
            row = idx // n_cols
            col = idx % n_cols
            
            ax = axes[row, col]
            
            # Get outlier bounds
            outliers_iqr, lower_bound, upper_bound, _, _, _ = self.detect_outliers_iqr(column)
            
            # Create histogram
            ax.hist(self.df[column].dropna(), bins=30, alpha=0.7, color='skyblue', label='Normal Data')
            
            # Highlight outliers
            if len(outliers_iqr) > 0:
                ax.hist(outliers_iqr[column], bins=30, alpha=0.7, color='red', label='Outliers')
            
            ax.set_title(f'{column} - Histogram')
            ax.set_xlabel(column)
            ax.set_ylabel('Frequency')
            ax.legend()
            
            # Add outlier bounds as vertical lines
            ax.axvline(lower_bound, color='orange', linestyle='--', alpha=0.7, label=f'Lower bound: {lower_bound:.2f}')
            ax.axvline(upper_bound, color='orange', linestyle='--', alpha=0.7, label=f'Upper bound: {upper_bound:.2f}')
        
        # Hide empty subplots
        for idx in range(len(numerical_columns), n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        save_plot(fig, 'outlier_histograms.png')
        plt.close()
        
        return fig
    
    def create_salary_outlier_analysis(self):
        """Create detailed salary outlier analysis."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Salary Outlier Analysis', fontsize=16)
        
        # Salary distribution by experience level
        experience_levels = self.df['experience_level'].unique()
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
        
        for i, level in enumerate(experience_levels):
            level_data = self.df[self.df['experience_level'] == level]['salary_usd']
            axes[0, 0].hist(level_data.dropna(), alpha=0.7, label=level, color=colors[i % len(colors)])
        
        axes[0, 0].set_title('Salary Distribution by Experience Level')
        axes[0, 0].set_xlabel('Salary (USD)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        
        # Salary box plot by experience level
        self.df.boxplot(column='salary_usd', by='experience_level', ax=axes[0, 1])
        axes[0, 1].set_title('Salary Distribution by Experience Level')
        axes[0, 1].set_xlabel('Experience Level')
        axes[0, 1].set_ylabel('Salary (USD)')
        
        # Salary vs remote ratio scatter plot
        axes[1, 0].scatter(self.df['remote_ratio'], self.df['salary_usd'], alpha=0.6)
        axes[1, 0].set_title('Salary vs Remote Ratio')
        axes[1, 0].set_xlabel('Remote Ratio (%)')
        axes[1, 0].set_ylabel('Salary (USD)')
        
        # Salary vs company size
        self.df.boxplot(column='salary_usd', by='company_size', ax=axes[1, 1])
        axes[1, 1].set_title('Salary Distribution by Company Size')
        axes[1, 1].set_xlabel('Company Size')
        axes[1, 1].set_ylabel('Salary (USD)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        save_plot(fig, 'salary_outlier_analysis.png')
        plt.close()
        
        return fig
    
    def generate_outlier_report(self):
        """Generate a comprehensive outlier analysis report."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        outlier_summary = self.analyze_numerical_outliers()
        
        report = []
        report.append("AI Job Market Dataset - Outlier Analysis Report")
        report.append("=" * 60)
        report.append("")
        
        report.append("1. OUTLIER DETECTION METHODS")
        report.append("-" * 30)
        report.append("• IQR Method: Uses Q1 - 1.5*IQR and Q3 + 1.5*IQR as bounds")
        report.append("• Z-score Method: Uses |z-score| > 3 as threshold")
        report.append("")
        
        report.append("2. OUTLIER SUMMARY BY COLUMN")
        report.append("-" * 30)
        
        for column, summary in outlier_summary.items():
            report.append(f"\n{column}:")
            report.append(f"  IQR outliers: {summary['iqr_outliers']} ({summary['iqr_outliers']/len(self.df)*100:.1f}%)")
            report.append(f"  Z-score outliers: {summary['zscore_outliers']} ({summary['zscore_outliers']/len(self.df)*100:.1f}%)")
            report.append(f"  IQR bounds: [{summary['iqr_bounds'][0]:.2f}, {summary['iqr_bounds'][1]:.2f}]")
            report.append(f"  Mean: {summary['mean']:.2f}")
            report.append(f"  Std: {summary['std']:.2f}")
        
        report.append("")
        report.append("3. RECOMMENDATIONS")
        report.append("-" * 30)
        report.append("• Review outliers in salary_usd to understand high-paying positions")
        report.append("• Check outliers in remote_ratio for unusual remote work arrangements")
        report.append("• Investigate outliers in years_experience for data quality issues")
        report.append("• Consider removing extreme outliers that may be data entry errors")
        
        report_text = "\n".join(report)
        save_report(report_text, 'outlier_analysis_report.txt')
        
        return report_text

def run_outlier_analysis(file_path=None):
    """Convenience function to run complete outlier analysis."""
    analyzer = OutlierAnalyzer()
    analyzer.load_data(file_path)
    analyzer.analyze_numerical_outliers()
    analyzer.create_outlier_plots()
    analyzer.create_salary_outlier_analysis()
    report = analyzer.generate_outlier_report()
    print("Outlier analysis completed. Check outputs/plots/ and outputs/reports/ for results.")
    return analyzer

if __name__ == "__main__":
    run_outlier_analysis() 