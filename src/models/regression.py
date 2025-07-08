import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from utils.config import get_config
from utils.helpers import save_plot, save_model, save_report

class SalaryPredictor:
    """Class to handle salary prediction using regression models."""
    
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
        
    def load_data(self, file_path=None):
        """Load the dataset for modeling."""
        if file_path is None:
            file_path = self.config['dataset']['cleaned_file']
        
        self.df = pd.read_csv(file_path)
        print(f"Loaded dataset with {len(self.df)} rows for salary prediction")
        return self.df
    
    def prepare_features(self):
        """Prepare features for salary prediction."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print("Preparing features for salary prediction...")
        
        # Select features for modeling
        feature_columns = [
            'experience_level', 'employment_type', 'company_size', 
            'remote_ratio', 'education_required', 'years_experience',
            'job_description_length', 'benefits_score'
        ]
        
        # Create feature matrix
        X = self.df[feature_columns].copy()
        y = self.df['salary_usd']
        
        # Encode categorical variables
        categorical_columns = ['experience_level', 'employment_type', 'company_size', 'education_required']
        
        for col in categorical_columns:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
        
        # Split the data
        random_state = self.config['models']['random_state']
        test_size = self.config['models']['test_size']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale numerical features
        numerical_columns = ['remote_ratio', 'years_experience', 'job_description_length', 'benefits_score']
        self.X_train[numerical_columns] = self.scaler.fit_transform(self.X_train[numerical_columns])
        self.X_test[numerical_columns] = self.scaler.transform(self.X_test[numerical_columns])
        
        print(f"Training set: {len(self.X_train)} samples")
        print(f"Test set: {len(self.X_test)} samples")
        print(f"Features: {list(X.columns)}")
        
        return X, y
    
    def train_models(self):
        """Train multiple regression models."""
        if self.X_train is None:
            raise ValueError("Features not prepared. Call prepare_features() first.")
        
        print("Training regression models...")
        
        # Define models to train
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=self.config['models']['random_state']),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=self.config['models']['random_state'])
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
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, self.X_train, self.y_train, 
                                      cv=self.config['models']['cv_folds'], 
                                      scoring='r2')
            
            results[name] = {
                'model': model,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            self.models[name] = model
            
            print(f"  R² Score: {r2:.4f}")
            print(f"  RMSE: ${rmse:,.0f}")
            print(f"  MAE: ${mae:,.0f}")
            print(f"  CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
        self.best_model = results[best_model_name]['model']
        self.best_score = results[best_model_name]['r2']
        
        print(f"\nBest model: {best_model_name} (R² = {self.best_score:.4f})")
        
        return results
    
    def create_model_comparison_plot(self, results):
        """Create comparison plot of model performance."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Salary Prediction Model Comparison', fontsize=16)
        
        # R² scores comparison
        model_names = list(results.keys())
        r2_scores = [results[name]['r2'] for name in model_names]
        
        axes[0, 0].bar(model_names, r2_scores, color='skyblue')
        axes[0, 0].set_title('R² Scores Comparison')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # RMSE comparison
        rmse_scores = [results[name]['rmse'] for name in model_names]
        
        axes[0, 1].bar(model_names, rmse_scores, color='lightcoral')
        axes[0, 1].set_title('RMSE Comparison')
        axes[0, 1].set_ylabel('RMSE (USD)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # MAE comparison
        mae_scores = [results[name]['mae'] for name in model_names]
        
        axes[1, 0].bar(model_names, mae_scores, color='lightgreen')
        axes[1, 0].set_title('MAE Comparison')
        axes[1, 0].set_ylabel('MAE (USD)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Cross-validation scores
        cv_means = [results[name]['cv_mean'] for name in model_names]
        cv_stds = [results[name]['cv_std'] for name in model_names]
        
        axes[1, 1].bar(model_names, cv_means, yerr=cv_stds, capsize=5, color='gold')
        axes[1, 1].set_title('Cross-Validation R² Scores')
        axes[1, 1].set_ylabel('CV R² Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        save_plot(fig, 'model_comparison.png')
        plt.close()
        
        return fig
    
    def create_prediction_plots(self):
        """Create prediction vs actual plots."""
        if self.best_model is None:
            raise ValueError("No best model available. Train models first.")
        
        # Get predictions from best model
        y_pred = self.best_model.predict(self.X_test)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Salary Prediction Results', fontsize=16)
        
        # Prediction vs Actual scatter plot
        axes[0, 0].scatter(self.y_test, y_pred, alpha=0.6)
        axes[0, 0].plot([self.y_test.min(), self.y_test.max()], 
                       [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Salary (USD)')
        axes[0, 0].set_ylabel('Predicted Salary (USD)')
        axes[0, 0].set_title('Predicted vs Actual Salary')
        
        # Residuals plot
        residuals = self.y_test - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Salary (USD)')
        axes[0, 1].set_ylabel('Residuals (USD)')
        axes[0, 1].set_title('Residuals Plot')
        
        # Residuals distribution
        axes[1, 0].hist(residuals, bins=30, alpha=0.7, color='skyblue')
        axes[1, 0].set_xlabel('Residuals (USD)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Residuals Distribution')
        
        # Prediction error distribution
        prediction_errors = np.abs(residuals)
        axes[1, 1].hist(prediction_errors, bins=30, alpha=0.7, color='lightcoral')
        axes[1, 1].set_xlabel('Absolute Prediction Error (USD)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Prediction Error Distribution')
        
        plt.tight_layout()
        save_plot(fig, 'prediction_analysis.png')
        plt.close()
        
        return fig
    
    def save_best_model(self):
        """Save the best performing model."""
        if self.best_model is None:
            raise ValueError("No best model available. Train models first.")
        
        save_model(self.best_model, 'salary_predictor.joblib')
        return 'salary_predictor.joblib'
    
    def generate_modeling_report(self, results):
        """Generate a comprehensive modeling report."""
        if self.best_model is None:
            raise ValueError("No best model available. Train models first.")
        
        report = []
        report.append("AI Job Market - Salary Prediction Modeling Report")
        report.append("=" * 60)
        report.append("")
        
        report.append("1. MODELING APPROACH")
        report.append("-" * 20)
        report.append("• Target variable: salary_usd")
        report.append("• Feature engineering: Label encoding for categorical variables")
        report.append("• Feature scaling: StandardScaler for numerical features")
        report.append("• Model selection: Multiple regression algorithms")
        report.append("• Evaluation: R², RMSE, MAE, and cross-validation")
        report.append("")
        
        report.append("2. MODEL PERFORMANCE COMPARISON")
        report.append("-" * 30)
        
        for name, metrics in results.items():
            report.append(f"\n{name}:")
            report.append(f"  R² Score: {metrics['r2']:.4f}")
            report.append(f"  RMSE: ${metrics['rmse']:,.0f}")
            report.append(f"  MAE: ${metrics['mae']:,.0f}")
            report.append(f"  CV R²: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std'] * 2:.4f})")
        
        report.append("")
        report.append("3. BEST MODEL")
        report.append("-" * 12)
        best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
        report.append(f"Model: {best_model_name}")
        report.append(f"R² Score: {results[best_model_name]['r2']:.4f}")
        report.append(f"RMSE: ${results[best_model_name]['rmse']:,.0f}")
        report.append(f"MAE: ${results[best_model_name]['mae']:,.0f}")
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
        report.append("5. RECOMMENDATIONS")
        report.append("-" * 15)
        report.append("• Use the best model for salary predictions")
        report.append("• Consider feature engineering for better performance")
        report.append("• Collect more data for improved accuracy")
        report.append("• Regular model retraining with new data")
        
        report_text = "\n".join(report)
        save_report(report_text, 'salary_prediction_report.txt')
        
        return report_text

def run_salary_prediction(input_file=None):
    """Convenience function to run complete salary prediction modeling."""
    predictor = SalaryPredictor()
    predictor.load_data(input_file)
    predictor.prepare_features()
    results = predictor.train_models()
    predictor.create_model_comparison_plot(results)
    predictor.create_prediction_plots()
    predictor.save_best_model()
    report = predictor.generate_modeling_report(results)
    print("Salary prediction modeling completed. Check outputs/plots/, outputs/models/, and outputs/reports/ for results.")
    return predictor

if __name__ == "__main__":
    run_salary_prediction() 