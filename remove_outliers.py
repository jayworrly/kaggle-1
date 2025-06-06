import pandas as pd
import numpy as np

# Load the dataset
file_path = 'data/ai_job_dataset.csv' # Assuming we start with the original data
df = pd.read_csv(file_path)

print(f"Original dataset shape: {df.shape}")

# Select numeric columns for outlier removal using IQR
numeric_cols = ['salary_usd', 'years_experience', 'remote_ratio', 'job_description_length', 'benefits_score']

# Calculate IQR and bounds for each numeric column
outlier_indices = np.array([], dtype=int)

for col in numeric_cols:
    if col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Find indices of outliers
        col_outlier_indices = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
        outlier_indices = np.concatenate((outlier_indices, col_outlier_indices))
        print(f"Found {len(col_outlier_indices)} outliers in '{col}'")
    else:
        print(f"Column {col} not found in the dataset. Skipping outlier removal for this column.")

# Get unique outlier indices
outlier_indices = np.unique(outlier_indices)

print(f"Total unique outliers found: {len(outlier_indices)}")

# Remove outliers
df_cleaned = df.drop(outlier_indices)

print(f"Cleaned dataset shape: {df_cleaned.shape}")

# Save the cleaned dataset
output_path = 'data/ai_job_dataset_cleaned.csv'
df_cleaned.to_csv(output_path, index=False)

print(f"Cleaned dataset saved to {output_path}") 