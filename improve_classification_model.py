import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import joblib

# Load the cleaned dataset
file_path = 'data/ai_job_dataset_cleaned.csv'
df = pd.read_csv(file_path)

# Define features (X) and target (y)
X = df.drop(['employment_type', 'job_id', 'salary_currency', 'employee_residence', 'company_location', 'company_name', 'posting_date', 'application_deadline', 'industry'], axis=1)
y = df['employment_type']

# Identify categorical and numeric features
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Create preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create a dictionary of models to evaluate
models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'GradientBoosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42)
}

# Define hyperparameters for tuning (simplified for demonstration)
param_grid = {
    'RandomForest': {'classifier__n_estimators': [50, 100]},
    'GradientBoosting': {'classifier__n_estimators': [50, 100]},
    'SVM': {'classifier__C': [0.1, 1]}
}

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

results = {}
best_model = None
best_accuracy = 0

print("Evaluating different classification models...")

for name, model in models.items():
    print(f"\nEvaluating {name}...")
    
    # Create pipeline with preprocessor and model
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    
    # Perform GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(pipeline, param_grid[name], cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters for {name}: {grid_search.best_params_}")
    
    # Evaluate the best model found by GridSearchCV
    y_pred = grid_search.best_estimator_.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    results[name] = {
        'best_params': grid_search.best_params_,
        'accuracy': accuracy,
        'report': report
    }
    
    print(f"Accuracy for {name}: {accuracy:.2f}")
    print(f"Classification Report for {name}:\n{report}")
    
    # Check if this is the best model so far
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = grid_search.best_estimator_

print("\nModel evaluation complete.")
print("\nSummary of Results:")
for name, result in results.items():
    print(f"{name}: Accuracy = {result['accuracy']:.2f}")

if best_model:
    print(f"\nBest performing model is: {best_model.steps[-1][0]} with Accuracy = {best_accuracy:.2f}")
    # Save the best model
    joblib.dump(best_model, 'best_classification_model.joblib')
    print("Best classification model saved as 'best_classification_model.joblib'")
else:
    print("No best model identified (possibly due to an error).") 