import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
import os

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

# Apply preprocessing
X_processed = preprocessor.fit_transform(df[numeric_features + categorical_features])

# Determine the optimal number of clusters using the Elbow Method (and Silhouette Score for consideration)
print("Determining the optimal number of clusters...")

distortions = []
silhouette_scores = []
K_range = range(2, 11) # Consider a range of cluster numbers

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_processed)
    distortions.append(kmeans.inertia_)
    
    # Calculate silhouette score (requires at least 2 clusters and data points >= n_clusters)
    if k > 1 and X_processed.shape[0] >= k:
        silhouette_avg = silhouette_score(X_processed, kmeans.labels_)
        silhouette_scores.append(silhouette_avg)
        print(f"For n_clusters = {k}, the average silhouette_score is : {silhouette_avg:.2f}")
    else:
        silhouette_scores.append(None) # Append None if silhouette score is not applicable

# Plot the Elbow Method graph
plt.figure(figsize=(10, 6))
plt.plot(K_range, distortions, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Distortion')
plt.xticks(K_range)
plt.grid(True)
plt.savefig('elbow_method.png')
plt.close()

print("Elbow method plot saved as 'elbow_method.png'")

# (Optional) Plot the Silhouette Scores
if any(score is not None for score in silhouette_scores):
    plt.figure(figsize=(10, 6))
    # Filter out None values for plotting
    silhouette_k_range = [K_range[i] for i, score in enumerate(silhouette_scores) if score is not None]
    valid_silhouette_scores = [score for score in silhouette_scores if score is not None]
    plt.plot(silhouette_k_range, valid_silhouette_scores, marker='o')
    plt.title('Silhouette Scores for Different K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.xticks(silhouette_k_range)
    plt.grid(True)
    plt.savefig('silhouette_scores.png')
    plt.close()
    print("Silhouette scores plot saved as 'silhouette_scores.png'")

# Based on the previous analysis, we determined k=2 is optimal. Proceed with clustering with k=2
optimal_k = 2
print(f"\nProceeding with K-means clustering with optimal_k = {optimal_k}")
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_processed)

# Save the clustered data (optional, but useful for further analysis)
df.to_csv('data/ai_job_dataset_clustered.csv', index=False) # Saving to a new file to keep the cleaned one intact
print("Clustered dataset saved as 'data/ai_job_dataset_clustered.csv'")

# Analyze cluster characteristics (basic summary)
print("\nCluster Analysis:")
print(df['cluster'].value_counts())

# You can add more detailed analysis here, like mean of features per cluster, etc.
# For example, to see the average salary per cluster:
print("\nAverage Salary by Cluster:")
print(df.groupby('cluster')['salary_usd'].mean())

# To see the distribution of experience levels by cluster:
print("\nExperience Level Distribution by Cluster:")
print(pd.crosstab(df['cluster'], df['experience_level'], normalize='index'))

# Save cluster assignments for further analysis
df[['job_id', 'cluster']].to_csv('cluster_analysis.csv', index=False)
print("Cluster assignments saved to 'cluster_analysis.csv'")

# Basic visualization of clusters (e.g., using first two principal components if needed, or pairplot of selected features)
# For simplicity, let's visualize salary vs years_experience with cluster colors
if 'salary_usd' in df.columns and 'years_experience' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='years_experience', y='salary_usd', hue='cluster', data=df, palette='viridis', legend='full')
    plt.title('Cluster Visualization (Salary vs Years Experience)')
    plt.xlabel('Years of Experience')
    plt.ylabel('Salary (USD)')
    plt.savefig('cluster_visualization.png')
    plt.close()
    print("Cluster visualization saved as 'cluster_visualization.png'") 