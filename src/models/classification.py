import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from utils.config import get_config
from utils.helpers import save_plot, save_model, save_report

class JobClassifier:
    """Class to handle job classification using machine learning models."""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.best_model = None
        self.best_score = 0
        self.target_column = 'experience_level'  # Can be changed to other categorical columns
        
    def load_data(self, file_path=None):
        """Load the dataset for classification."""
        if file_path is None:
            file_path = self.config['dataset']['cleaned_file']
        
        self.df = pd.read_csv(file_path)
        print(f"Loaded dataset with {len(self.df)} rows for job classification")
        return self.df
    
    def prepare_features(self, target_column='experience_level'):
        """Prepare features for job classification."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        self.target_column = target_column
        print(f"Preparing features for {target_column} classification...")
        
        # Select features for modeling
        feature_columns = [
            'salary_usd', 'employment_type', 'company_size', 
            'remote_ratio', 'education_required', 'years_experience',
            'job_description_length', 'benefits_score'
        ]
        
        # Create feature matrix
        X = self.df[feature_columns].copy()
        y = self.df[target_column]
        
        # Encode categorical variables
        categorical_columns = ['employment_type', 'company_size', 'education_required']
        
        for col in categorical_columns:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
        
        # Encode target variable
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y.astype(str))
        self.label_encoders[target_column] = target_encoder
        
        # Split the data
        random_state = self.config['models']['random_state']
        test_size = self.config['models']['test_size']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale numerical features
        numerical_columns = ['salary_usd', 'remote_ratio', 'years_experience', 'job_description_length', 'benefits_score']
        self.X_train[numerical_columns] = self.scaler.fit_transform(self.X_train[numerical_columns])
        self.X_test[numerical_columns] = self.scaler.transform(self.X_test[numerical_columns])
        
        print(f"Training set: {len(self.X_train)} samples")
        print(f"Test set: {len(self.X_test)} samples")
        print(f"Features: {list(X.columns)}")
        print(f"Target classes: {target_encoder.classes_}")
        
        return X, y
    
    def train_models(self):
        """Train multiple classification models."""
        if self.X_train is None:
            raise ValueError("Features not prepared. Call prepare_features() first.")
        
        print("Training classification models...")
        
        # Define models to train
        models = {
            'Logistic Regression': LogisticRegression(random_state=self.config['models']['random_state'], max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=self.config['models']['random_state']),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=self.config['models']['random_state']),
            'SVM': SVC(random_state=self.config['models']['random_state'])
        }
        
        # Train and evaluate each model
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train the model
            model.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            
            # Cross-validation score
            cv_scores = cross_val_score(model, self.X_train, self.y_train, 
                                      cv=self.config['models']['cv_folds'], 
                                      scoring='accuracy')
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'f1_score': f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred
            }
            
            self.models[name] = model
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  F1 Score: {f1:.4f}")
            print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
        self.best_model = results[best_model_name]['model']
        self.best_score = results[best_model_name]['accuracy']
        
        print(f"\nBest model: {best_model_name} (Accuracy = {self.best_score:.4f})")
        
        return results
    
    def hyperparameter_tuning(self, model_name='Random Forest'):
        """Perform hyperparameter tuning for the best model."""
        if self.X_train is None:
            raise ValueError("Features not prepared. Call prepare_features() first.")
        
        print(f"Performing hyperparameter tuning for {model_name}...")
        
        if model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = RandomForestClassifier(random_state=self.config['models']['random_state'])
        elif model_name == 'Gradient Boosting':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            }
            model = GradientBoostingClassifier(random_state=self.config['models']['random_state'])
        else:
            print(f"Hyperparameter tuning not implemented for {model_name}")
            return None
        
        # Perform grid search
        grid_search = GridSearchCV(
            model, param_grid, cv=self.config['models']['cv_folds'], 
            scoring='accuracy', n_jobs=-1, verbose=1
        )
        grid_search.fit(self.X_train, self.y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        # Update best model
        self.best_model = grid_search.best_estimator_
        self.models[f'{model_name} (Tuned)'] = self.best_model
        
        return grid_search
    
    def create_model_comparison_plot(self, results):
        """Create comparison plot of model performance."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{self.target_column} Classification Model Comparison', fontsize=16)
        
        # Accuracy comparison
        model_names = list(results.keys())
        accuracy_scores = [results[name]['accuracy'] for name in model_names]
        
        axes[0, 0].bar(model_names, accuracy_scores, color='skyblue')
        axes[0, 0].set_title('Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # F1 Score comparison
        f1_scores = [results[name]['f1_score'] for name in model_names]
        
        axes[0, 1].bar(model_names, f1_scores, color='lightcoral')
        axes[0, 1].set_title('F1 Score Comparison')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Cross-validation scores
        cv_means = [results[name]['cv_mean'] for name in model_names]
        cv_stds = [results[name]['cv_std'] for name in model_names]
        
        axes[1, 0].bar(model_names, cv_means, yerr=cv_stds, capsize=5, color='lightgreen')
        axes[1, 0].set_title('Cross-Validation Accuracy Scores')
        axes[1, 0].set_ylabel('CV Accuracy Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Model comparison summary
        comparison_data = pd.DataFrame({
            'Model': model_names,
            'Accuracy': accuracy_scores,
            'F1_Score': f1_scores,
            'CV_Accuracy': cv_means
        })
        
        axes[1, 1].axis('tight')
        axes[1, 1].axis('off')
        table = axes[1, 1].table(cellText=comparison_data.values, 
                                colLabels=comparison_data.columns,
                                cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        axes[1, 1].set_title('Model Performance Summary')
        
        plt.tight_layout()
        save_plot(fig, f'{self.target_column}_model_comparison.png')
        plt.close()
        
        return fig
    
    def create_confusion_matrix_plot(self, results):
        """Create confusion matrix plots for all models."""
        n_models = len(results)
        n_cols = min(2, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 6 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(f'{self.target_column} Classification - Confusion Matrices', fontsize=16)
        
        for idx, (name, result) in enumerate(results.items()):
            row = idx // n_cols
            col = idx % n_cols
            
            # Create confusion matrix
            cm = confusion_matrix(self.y_test, result['predictions'])
            
            # Plot confusion matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=self.label_encoders[self.target_column].classes_,
                       yticklabels=self.label_encoders[self.target_column].classes_,
                       ax=axes[row, col])
            axes[row, col].set_title(f'{name}\nAccuracy: {result["accuracy"]:.3f}')
            axes[row, col].set_xlabel('Predicted')
            axes[row, col].set_ylabel('Actual')
        
        # Hide empty subplots
        for idx in range(n_models, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        save_plot(fig, f'{self.target_column}_confusion_matrices.png')
        plt.close()
        
        return fig
    
    def create_classification_report_plot(self, results):
        """Create detailed classification report visualization."""
        if self.best_model is None:
            raise ValueError("No best model available. Train models first.")
        
        # Get predictions from best model
        y_pred = self.best_model.predict(self.X_test)
        
        # Generate classification report
        report = classification_report(self.y_test, y_pred, 
                                     target_names=self.label_encoders[self.target_column].classes_,
                                     output_dict=True)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'{self.target_column} Classification - Detailed Report', fontsize=16)
        
        # Precision, Recall, F1-score
        metrics_df = pd.DataFrame(report).transpose()
        metrics_df = metrics_df.drop('support', axis=1)
        metrics_df = metrics_df.drop('accuracy', axis=0)
        
        metrics_df.plot(kind='bar', ax=axes[0])
        axes[0].set_title('Precision, Recall, and F1-Score by Class')
        axes[0].set_ylabel('Score')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].legend()
        
        # Support (class distribution)
        support = [report[cls]['support'] for cls in self.label_encoders[self.target_column].classes_]
        axes[1].bar(self.label_encoders[self.target_column].classes_, support, color='lightcoral')
        axes[1].set_title('Class Distribution in Test Set')
        axes[1].set_ylabel('Number of Samples')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        save_plot(fig, f'{self.target_column}_classification_report.png')
        plt.close()
        
        return fig
    
    def save_best_model(self):
        """Save the best performing model."""
        if self.best_model is None:
            raise ValueError("No best model available. Train models first.")
        
        save_model(self.best_model, f'{self.target_column}_classifier.joblib')
        return f'{self.target_column}_classifier.joblib'
    
    def generate_classification_report(self, results):
        """Generate a comprehensive classification report."""
        if self.best_model is None:
            raise ValueError("No best model available. Train models first.")
        
        report = []
        report.append(f"AI Job Market - {self.target_column} Classification Report")
        report.append("=" * 60)
        report.append("")
        
        report.append("1. CLASSIFICATION APPROACH")
        report.append("-" * 25)
        report.append(f"• Target variable: {self.target_column}")
        report.append("• Feature engineering: Label encoding for categorical variables")
        report.append("• Feature scaling: StandardScaler for numerical features")
        report.append("• Model selection: Multiple classification algorithms")
        report.append("• Evaluation: Accuracy, F1-score, and cross-validation")
        report.append("")
        
        report.append("2. MODEL PERFORMANCE COMPARISON")
        report.append("-" * 30)
        
        for name, metrics in results.items():
            report.append(f"\n{name}:")
            report.append(f"  Accuracy: {metrics['accuracy']:.4f}")
            report.append(f"  F1 Score: {metrics['f1_score']:.4f}")
            report.append(f"  CV Accuracy: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std'] * 2:.4f})")
        
        report.append("")
        report.append("3. BEST MODEL")
        report.append("-" * 12)
        best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
        report.append(f"Model: {best_model_name}")
        report.append(f"Accuracy: {results[best_model_name]['accuracy']:.4f}")
        report.append(f"F1 Score: {results[best_model_name]['f1_score']:.4f}")
        report.append("")
        
        report.append("4. FEATURE IMPORTANCE")
        report.append("-" * 20)
        if hasattr(self.best_model, 'feature_importances_'):
            feature_importance = self.best_model.feature_importances_
            feature_names = self.X_train.columns
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importance
            }).sort_values('Importance', ascending=False)
            
            for _, row in importance_df.iterrows():
                report.append(f"  {row['Feature']}: {row['Importance']:.4f}")
        else:
            report.append("  Feature importance not available for this model type")
        
        report.append("")
        report.append("5. CLASSIFICATION DETAILS")
        report.append("-" * 25)
        y_pred = self.best_model.predict(self.X_test)
        class_report = classification_report(self.y_test, y_pred, 
                                           target_names=self.label_encoders[self.target_column].classes_)
        report.append(class_report)
        
        report.append("6. RECOMMENDATIONS")
        report.append("-" * 15)
        report.append("• Use the best model for job classification")
        report.append("• Consider feature engineering for better performance")
        report.append("• Collect more data for improved accuracy")
        report.append("• Regular model retraining with new data")
        
        report_text = "\n".join(report)
        save_report(report_text, f'{self.target_column}_classification_report.txt')
        
        return report_text

def run_job_classification(input_file=None, target_column='experience_level'):
    """Convenience function to run complete job classification modeling."""
    classifier = JobClassifier()
    classifier.load_data(input_file)
    classifier.prepare_features(target_column)
    results = classifier.train_models()
    classifier.create_model_comparison_plot(results)
    classifier.create_confusion_matrix_plot(results)
    classifier.create_classification_report_plot(results)
    classifier.save_best_model()
    report = classifier.generate_classification_report(results)
    print(f"Job classification modeling completed for {target_column}. Check outputs/plots/, outputs/models/, and outputs/reports/ for results.")
    return classifier

if __name__ == "__main__":
    run_job_classification() 