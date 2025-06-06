import pandas as pd

# Load the dataset
file_path = 'data/ai_job_dataset_cleaned.csv'
df = pd.read_csv(file_path)

# Display basic information
print("Dataset Information:")
df.info()

# Display the first few rows
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Display descriptive statistics
print("\nDescriptive Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check for duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Check unique values for categorical columns (example)
print("\nUnique values in key categorical columns:")
for col in ['experience_level', 'employment_type', 'company_size', 'education_required']:
    if col in df.columns:
        print(f"{col}: {df[col].unique()}")

print("\nEDA and cleaning check complete.") 