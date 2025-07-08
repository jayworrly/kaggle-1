# AI Job Market Analysis

A comprehensive data science project analyzing the AI job market using machine learning techniques for salary prediction, job classification, and market segmentation.

## 🎯 Project Overview

This project analyzes AI job market data to provide insights into:
- **Salary Prediction**: Predict salaries based on job characteristics
- **Job Classification**: Categorize jobs by experience level, company size, etc.
- **Market Segmentation**: Identify distinct job market clusters
- **Trend Analysis**: Understand market dynamics and patterns

## 📊 Dataset

**Source**: Kaggle AI Job Market Dataset  
**Size**: 15,000 job postings  
**Features**: 19 columns including salary, experience, location, company details, and job requirements

### Key Columns
- `salary_usd`: Annual salary in USD (target for regression)
- `experience_level`: EN (Entry), MI (Mid), SE (Senior), EX (Executive)
- `employment_type`: FT, PT, CT, FL
- `company_size`: S (Small), M (Medium), L (Large)
- `remote_ratio`: 0, 50, 100 (percentage remote work)
- `job_title`, `company_location`, `employee_residence`

## 🏗️ Project Structure

```
├── src/                    # Source code (modular architecture)
│   ├── data/              # Data processing modules
│   │   ├── loader.py      # Database operations and data loading
│   │   └── cleaner.py     # Data cleaning and preprocessing
│   ├── analysis/          # Analysis modules
│   │   ├── eda.py         # Exploratory data analysis
│   │   ├── outliers.py    # Outlier detection and analysis
│   │   └── cleaning.py    # Advanced data cleaning
│   ├── models/            # Machine learning modules
│   │   ├── regression.py  # Salary prediction models
│   │   ├── classification.py # Job classification models
│   │   └── clustering.py  # Market segmentation models
│   └── utils/             # Utility modules
│       ├── config.py      # Configuration management
│       └── helpers.py     # Helper functions
├── scripts/               # Pipeline scripts
│   ├── run_pipeline.py    # Complete data pipeline
│   ├── run_analysis.py    # Analysis pipeline
│   └── run_modeling.py    # Modeling pipeline
├── config/                # Configuration files
│   └── config.yaml        # Main configuration
├── outputs/               # Generated outputs
│   ├── plots/            # Visualizations
│   ├── models/           # Trained models
│   └── reports/          # Analysis reports
├── data/                  # Data files
├── tests/                 # Unit tests
├── run_modeling.py        # Main launcher script
└── main.py               # CLI entry point
```

## ✨ Features

### 🔄 Modular Architecture
- **Clean separation** of data processing, analysis, and modeling
- **Reusable components** with consistent interfaces
- **Configuration-driven** with YAML config and environment variables
- **Comprehensive error handling** and logging

### 📈 Advanced Analytics
- **Exploratory Data Analysis**: Distribution analysis, correlation matrices, trend visualization
- **Outlier Detection**: Statistical and ML-based outlier identification
- **Data Quality Assessment**: Missing values, duplicates, data type validation

### 🤖 Machine Learning Pipeline
- **Regression**: Salary prediction using Multiple algorithms (Linear, Ridge, Lasso, Random Forest, Gradient Boosting)
- **Classification**: Job categorization with high accuracy (99%+ accuracy achieved)
- **Clustering**: Market segmentation using K-Means, DBSCAN, Agglomerative, Gaussian Mixture

### 📊 Comprehensive Reporting
- **Automated visualizations** with professional plots
- **Detailed reports** for each analysis phase
- **Model performance metrics** and comparisons
- **Business insights** and recommendations

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Install dependencies
pip install -r requirements.txt

# Set up database credentials (optional)
cp .env.template .env
# Edit .env with your database credentials
```

### 2. Run Complete Analysis
```bash
# Run everything (recommended for first-time users)
python run_modeling.py --all

# Or run specific components
python run_modeling.py --regression
python run_modeling.py --classification  
python run_modeling.py --clustering
```

### 3. Explore Results
- **Plots**: `outputs/plots/` - Visualizations and charts
- **Models**: `outputs/models/` - Trained ML models (.joblib files)
- **Reports**: `outputs/reports/` - Detailed analysis reports

## 📋 Pipeline Scripts

### Data Pipeline
```bash
python scripts/run_pipeline.py    # Complete data processing
```

### Analysis Pipeline  
```bash
python scripts/run_analysis.py    # EDA, outliers, cleaning
```

### Modeling Pipeline
```bash
python scripts/run_modeling.py    # ML models and predictions
```

## 📈 Results Summary

### 🎯 Model Performance
- **Salary Prediction**: R² = 0.66 (Gradient Boosting)
- **Job Classification**: 100% Accuracy (Random Forest)
- **Market Segmentation**: 2 optimal clusters identified

### 💡 Key Insights
1. **Experience level** is the strongest predictor of salary
2. **Remote work ratio** significantly impacts compensation
3. **Company size** correlates with salary ranges
4. **Geographic location** affects market dynamics
5. **Job market** segments into distinct clusters with different characteristics

## 🛠️ Technologies Used

- **Python 3.8+**
- **Data Science**: pandas, numpy, scipy
- **Machine Learning**: scikit-learn  
- **Visualization**: matplotlib, seaborn
- **Database**: SQLite
- **Configuration**: PyYAML, python-dotenv
- **Testing**: pytest

## 📁 Key Files

- `run_modeling.py` - Main launcher script
- `src/models/regression.py` - Salary prediction models
- `src/models/classification.py` - Job classification models  
- `src/models/clustering.py` - Market segmentation models
- `config/config.yaml` - Configuration settings
- `requirements.txt` - Python dependencies

## 🎓 Usage Examples

### Custom Target Classification
```bash
python run_modeling.py --classification --target-column company_size
```

### Specific Cluster Count
```bash
python run_modeling.py --clustering --n-clusters 5
```

### Individual Model Training
```bash
python -c "from src.models.regression import run_salary_prediction; run_salary_prediction()"
```

## 🔍 What's Generated

After running the pipeline, you'll have:

### 📊 Visualizations
- Model performance comparisons
- Prediction vs actual plots  
- Confusion matrices
- Cluster visualizations
- Feature importance plots
- Data distribution charts

### 🤖 Trained Models
- Salary predictor (regression)
- Experience level classifier
- Job market clusterer

### 📋 Detailed Reports
- Model performance metrics
- Feature importance analysis
- Cluster characteristics
- Business recommendations

## ⚙️ Configuration

The project uses a centralized configuration system:

- `config/config.yaml` - Main settings
- `.env` - Environment-specific variables (database credentials, API keys)
- Command-line arguments for runtime customization

## 🧪 Testing

```bash
# Run unit tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_models.py
```

## 📈 Next Steps

- [ ] Web interface for interactive analysis
- [ ] Real-time data pipeline integration
- [ ] Advanced feature engineering
- [ ] Deep learning models
- [ ] API for model serving
- [ ] Dashboard development

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Built with ❤️ for data science and AI job market analysis** 