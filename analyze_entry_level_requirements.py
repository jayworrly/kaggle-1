import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the data
df = pd.read_csv('data/ai_job_dataset_cleaned.csv')

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

# Filter for Cluster 1 (Entry/Mid-level positions)
cluster1_df = df[df['cluster'] == 1]

# Create directory for plots
import os
if not os.path.exists('entry_level_analysis'):
    os.makedirs('entry_level_analysis')

# 1. Analyze Required Skills and Their Frequency
def analyze_skills():
    # Get all skills
    all_skills = []
    for skills in cluster1_df['required_skills'].dropna():
        all_skills.extend([s.strip() for s in skills.split(',')])
    
    # Count skill frequencies
    skill_counts = Counter(all_skills)
    
    # Get top 20 skills
    top_skills = dict(skill_counts.most_common(20))
    
    # Create bar plot
    plt.figure(figsize=(15, 8))
    plt.bar(top_skills.keys(), top_skills.values())
    plt.xticks(rotation=45, ha='right')
    plt.title('Top 20 Required Skills for Entry/Mid-Level AI Positions')
    plt.tight_layout()
    plt.savefig('entry_level_analysis/top_skills.png')
    plt.close()
    
    return top_skills

# 2. Analyze Education Requirements
def analyze_education():
    edu_counts = cluster1_df['education_required'].value_counts()
    
    plt.figure(figsize=(10, 6))
    edu_counts.plot(kind='bar')
    plt.title('Education Requirements for Entry/Mid-Level Positions')
    plt.xlabel('Education Level')
    plt.ylabel('Number of Positions')
    plt.tight_layout()
    plt.savefig('entry_level_analysis/education_requirements.png')
    plt.close()
    
    return edu_counts

# 3. Analyze Experience Requirements
def analyze_experience():
    plt.figure(figsize=(10, 6))
    sns.histplot(data=cluster1_df, x='years_experience', bins=20)
    plt.title('Years of Experience Distribution')
    plt.xlabel('Years of Experience')
    plt.ylabel('Number of Positions')
    plt.savefig('entry_level_analysis/experience_distribution.png')
    plt.close()
    
    return cluster1_df['years_experience'].describe()

# 5. Analyze Salary Distribution
def analyze_salary():
    plt.figure(figsize=(10, 6))
    sns.histplot(data=cluster1_df, x='salary_usd', bins=30)
    plt.title('Salary Distribution for Entry/Mid-Level Positions')
    plt.xlabel('Salary (USD)')
    plt.ylabel('Number of Positions')
    plt.savefig('entry_level_analysis/salary_distribution.png')
    plt.close()
    
    return cluster1_df['salary_usd'].describe()

# Run all analyses
print("Analyzing Entry/Mid-Level Position Requirements...")
print("\n1. Required Skills Analysis:")
top_skills = analyze_skills()
print("\nTop 20 Required Skills:")
for skill, count in top_skills.items():
    print(f"{skill}: {count} positions")

print("\n2. Education Requirements:")
edu_counts = analyze_education()
print("\nEducation Level Distribution:")
print(edu_counts)

print("\n3. Experience Requirements:")
exp_stats = analyze_experience()
print("\nExperience Statistics:")
print(exp_stats)

print("\n5. Salary Analysis:")
salary_stats = analyze_salary()
print("\nSalary Statistics:")
print(salary_stats)

# Save comprehensive analysis to file
with open('entry_level_analysis/interview_preparation_guide.txt', 'w') as f:
    f.write("Entry/Mid-Level AI Position Interview Preparation Guide\n")
    f.write("=" * 50 + "\n\n")
    
    f.write("1. Required Technical Skills\n")
    f.write("-" * 30 + "\n")
    f.write("Focus on mastering these core skills:\n")
    for skill, count in top_skills.items():
        f.write(f"- {skill}: Required in {count} positions\n")
    
    f.write("\n2. Education Requirements\n")
    f.write("-" * 30 + "\n")
    f.write("Typical education requirements:\n")
    for edu, count in edu_counts.items():
        f.write(f"- {edu}: {count} positions\n")
    
    f.write("\n3. Experience Requirements\n")
    f.write("-" * 30 + "\n")
    f.write(f"Average years of experience: {exp_stats['mean']:.1f} years\n")
    f.write(f"Experience range: {exp_stats['min']:.1f} - {exp_stats['max']:.1f} years\n")
    
    f.write("\n5. Salary Expectations\n")
    f.write("-" * 30 + "\n")
    f.write(f"Average salary: ${salary_stats['mean']:,.2f}\n")
    f.write(f"Salary range: ${salary_stats['min']:,.2f} - ${salary_stats['max']:,.2f}\n")
    
    f.write("\n6. Interview Preparation Tips\n")
    f.write("-" * 30 + "\n")
    f.write("1. Technical Preparation:\n")
    f.write("   - Focus on mastering Python and SQL as they are the most commonly required skills\n")
    f.write("   - Build projects using TensorFlow and PyTorch\n")
    f.write("   - Practice with real-world datasets and problems\n\n")
    
    f.write("2. Soft Skills Development:\n")
    f.write("   - Work on communication and presentation skills\n")
    f.write("   - Practice explaining technical concepts clearly\n")
    f.write("   - Develop problem-solving and analytical thinking abilities\n\n")
    
    f.write("3. Portfolio Development:\n")
    f.write("   - Create a GitHub portfolio with relevant projects\n")
    f.write("   - Document your work and thought processes\n")
    f.write("   - Include both individual and team projects\n\n")
    
    f.write("4. Interview Strategy:\n")
    f.write("   - Prepare for both technical and behavioral questions\n")
    f.write("   - Practice coding problems and system design questions\n")
    f.write("   - Be ready to discuss your projects in detail\n")
    f.write("   - Research the company and its AI initiatives\n") 