# Global AI Job Market and Salary Trends 2025 Analysis

This project analyzes the Global AI Job Market and Salary Trends 2025 dataset from Kaggle to understand key trends, build predictive models, and identify distinct job market segments.

## Project Structure

- `data/`: Intended to contain the raw and cleaned dataset (excluded from version control by `.gitignore`).
- Python scripts (`.py` files): Located in the root directory for data downloading, processing, modeling, and analysis.
- `.gitignore`: Specifies files and directories to be excluded from version control.
- `requirements.txt`: Lists the project dependencies.

Directories for generated outputs (excluded by `.gitignore`):
- `outlier_plots/`: Contains outlier visualization plots.
- `cluster_analysis_plots/`: Contains cluster analysis plots.
- `entry_level_analysis/`: Contains entry-level analysis plots and guides.

Other generated files (excluded by `.gitignore`):
- `.joblib`: Saved model files.
- `.png`, `.csv`, `.txt`: Generated plots and analysis results.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/jayworrly/kaggle-1.git
    cd kaggle-1
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Kaggle API Setup:**
    - Go to your Kaggle account settings and download your `kaggle.json` file.
    - Place the `kaggle.json` file in `C:\Users\<Your_Username>\.kaggle\` (on Windows) or `~/.kaggle/` (on macOS/Linux).

4.  **PostgreSQL Setup (Optional, for database import):**
    - Install PostgreSQL if you haven't already.
    - Update the database connection details in `create_database.py` and `import_to_postgres.py` with your credentials.
    - Run `python create_database.py` to create the database.

## Usage

To reproduce the analysis and results, you need to run the scripts sequentially:

1.  **Download the dataset:**
    ```bash
    python download_dataset.py
    ```
    This will download the dataset into the `data/` directory.

2.  **Import data to PostgreSQL (Optional):**
    ```bash
    python import_to_postgres.py
    ```

3.  **Perform EDA and Cleaning:**
    ```bash
    python eda_check_cleaning.py
    python visualize_outliers.py
    python remove_outliers.py
    ```
    This will generate the cleaned dataset in `data/ai_job_dataset_cleaned.csv` and outlier plots in `outlier_plots/`.

4.  **Build and Evaluate Models:**
    - Regression Model:
      ```bash
      python build_regression_model.py
      ```
    - Classification Model:
      ```bash
      python build_classification_model.py
      python improve_classification_model.py
      ```
    - Clustering Model:
      ```bash
      python build_clustering_model.py
      python analyze_cluster_features.py
      python analyze_entry_level_requirements.py
      ```
    These scripts will generate model files (`.joblib`), cluster analysis results (`cluster_analysis.csv`, `cluster_differences_analysis.txt`, plots in `cluster_analysis_plots/`), and entry-level analysis results (plots and `interview_preparation_guide.txt` in `entry_level_analysis/`).

## Analysis and Results

- The analysis scripts generate various plots and text files summarizing the findings, which are saved in their respective directories (`outlier_plots/`, `cluster_analysis_plots/`, `entry_level_analysis/`).
- Key analysis results are also printed to the console during script execution. 