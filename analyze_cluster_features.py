import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the cleaned dataset
file_path = 'data/ai_job_dataset_cleaned.csv'
df = pd.read_csv(file_path)

# Select features for clustering
numeric_features = ['salary_usd', 'years_experience', 'remote_ratio', 'job_description_length', 'benefits_score']
categorical_features = ['experience_level', 'employment_type', 'company_size', 'education_required']

# Create preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Prepare the data
X = df[numeric_features + categorical_features]
X_processed = preprocessor.fit_transform(X)

# Perform clustering (assuming 2 clusters based on previous analysis)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_processed)

# Create a directory for plots
import os
if not os.path.exists('cluster_analysis_plots'):
    os.makedirs('cluster_analysis_plots')

# Function to create box plots for numeric features
def plot_numeric_features(df, feature):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='cluster', y=feature, data=df)
    plt.title(f'{feature} Distribution by Cluster')
    plt.savefig(f'cluster_analysis_plots/{feature}_distribution.png')
    plt.close()

# Function to create bar plots for categorical features
def plot_categorical_features(df, feature):
    plt.figure(figsize=(12, 6))
    cluster_cat = pd.crosstab(df['cluster'], df[feature], normalize='index') * 100
    cluster_cat.plot(kind='bar', stacked=True)
    plt.title(f'{feature} Distribution by Cluster')
    plt.ylabel('Percentage')
    plt.legend(title=feature, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'cluster_analysis_plots/{feature}_distribution.png')
    plt.close()

# Analyze numeric features
numeric_features = ['salary_usd', 'years_experience', 'remote_ratio', 'job_description_length', 'benefits_score']
print("\nNumeric Feature Analysis:")
print("-" * 50)

for feature in numeric_features:
    # Create box plot
    plot_numeric_features(df, feature)
    
    # Calculate statistics
    cluster_stats = df.groupby('cluster')[feature].agg(['mean', 'median', 'std'])
    print(f"\n{feature} Statistics by Cluster:")
    print(cluster_stats)
    
    # Perform t-test
    cluster0 = df[df['cluster'] == 0][feature]
    cluster1 = df[df['cluster'] == 1][feature]
    t_stat, p_value = stats.ttest_ind(cluster0, cluster1)
    print(f"T-test p-value: {p_value:.4f}")

# Analyze categorical features
categorical_features = ['experience_level', 'employment_type', 'company_size', 'education_required']
print("\nCategorical Feature Analysis:")
print("-" * 50)

for feature in categorical_features:
    # Create bar plot
    plot_categorical_features(df, feature)
    
    # Calculate percentages
    print(f"\n{feature} Distribution by Cluster:")
    print(pd.crosstab(df['cluster'], df[feature], normalize='index') * 100)
    
    # Perform chi-square test
    chi2, p_value = stats.chi2_contingency(pd.crosstab(df['cluster'], df[feature]))[0:2]
    print(f"Chi-square test p-value: {p_value:.4f}")

# Analyze job titles
print("\nJob Title Analysis:")
print("-" * 50)
for cluster in [0, 1]:
    print(f"\nTop 10 Job Titles in Cluster {cluster}:")
    print(df[df['cluster'] == cluster]['job_title'].value_counts().head(10))

# Analyze required skills
print("\nRequired Skills Analysis:")
print("-" * 50)
for cluster in [0, 1]:
    # Get all skills for the cluster
    skills = df[df['cluster'] == cluster]['required_skills'].str.split(',').explode()
    skills = skills.str.strip()
    
    print(f"\nTop 10 Required Skills in Cluster {cluster}:")
    print(skills.value_counts().head(10))

# Create a summary of key differences
print("\nKey Differences Between Clusters:")
print("-" * 50)

# Calculate effect sizes for numeric features
for feature in numeric_features:
    cluster0_mean = df[df['cluster'] == 0][feature].mean()
    cluster1_mean = df[df['cluster'] == 1][feature].mean()
    diff = cluster1_mean - cluster0_mean
    print(f"\n{feature}:")
    print(f"Difference between clusters: {diff:.2f}")
    print(f"Cluster 0 mean: {cluster0_mean:.2f}")
    print(f"Cluster 1 mean: {cluster1_mean:.2f}")

# Save the analysis to a text file
with open('cluster_differences_analysis.txt', 'w') as f:
    f.write("Cluster Analysis Summary\n")
    f.write("=" * 50 + "\n\n")
    
    f.write("Numeric Features Analysis:\n")
    f.write("-" * 30 + "\n")
    for feature in numeric_features:
        stats = df.groupby('cluster')[feature].agg(['mean', 'median', 'std'])
        f.write(f"\n{feature}:\n{stats}\n")
    
    f.write("\nCategorical Features Analysis:\n")
    f.write("-" * 30 + "\n")
    for feature in categorical_features:
        dist = pd.crosstab(df['cluster'], df[feature], normalize='index') * 100
        f.write(f"\n{feature}:\n{dist}\n")
    
    f.write("\nTop Job Titles by Cluster:\n")
    f.write("-" * 30 + "\n")
    for cluster in [0, 1]:
        f.write(f"\nCluster {cluster}:\n")
        f.write(str(df[df['cluster'] == cluster]['job_title'].value_counts().head(10)))
        f.write("\n") 