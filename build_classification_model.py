import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Load the cleaned dataset
file_path = 'data/ai_job_dataset_cleaned.csv'
df = pd.read_csv(file_path)

# Define features (X) and target (y)
# We will try to predict 'employment_type'
# Exclude columns not suitable for features or the target itself
X = df.drop(['employment_type', 'job_id', 'salary_currency', 'employee_residence', 'company_location', 'company_name', 'posting_date', 'application_deadline', 'industry'], axis=1)
y = df['employment_type']

# Identify categorical and numeric features
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Create preprocessing pipelines for numeric and categorical features
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create a column transformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create the full classification pipeline
# Using Logistic Regression as a starting point
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train the model
print("Training the classification model...")
model.fit(X_train, y_train)
print("Training complete.")

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"\nModel Evaluation:")
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(report)

# Save the trained model
joblib.dump(model, 'classification_model.joblib')
print("Trained classification model saved as 'classification_model.joblib'") 