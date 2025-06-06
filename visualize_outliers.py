import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the dataset
file_path = 'data/ai_job_dataset.csv' # Assuming we start with the original data to visualize outliers before removal
df = pd.read_csv(file_path)

# Select numeric columns for outlier visualization
numeric_cols = ['salary_usd', 'years_experience', 'remote_ratio', 'job_description_length', 'benefits_score']

# Create a directory for outlier plots
plots_dir = 'outlier_plots'
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

print(f"Visualizing outliers for numeric columns: {numeric_cols}")

# Visualize outliers using box plots
for col in numeric_cols:
    if col in df.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=df[col])
        plt.title(f'Box Plot of {col}')
        plt.xlabel(col)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{col}_boxplot.png'))
        plt.close()
        print(f"Generated box plot for {col}")
    else:
        print(f"Column {col} not found in the dataset.")

print("Outlier visualization complete. Box plots are saved in the 'outlier_plots' directory.") 