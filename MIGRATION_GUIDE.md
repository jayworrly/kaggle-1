# Migration Guide: From Scripts to Modular Structure

This guide explains the changes made during the refactoring and how to migrate from the old script-based approach to the new modular structure.

## What Changed

### Before (Script-based)
```
├── download_dataset.py
├── create_database.py
├── import_to_postgres.py
├── eda_check_cleaning.py
├── visualize_outliers.py
├── remove_outliers.py
├── build_regression_model.py
├── build_classification_model.py
├── build_clustering_model.py
└── ... (many more individual scripts)
```

### After (Modular)
```
├── src/
│   ├── data/
│   │   ├── downloader.py      # Replaces download_dataset.py
│   │   └── database.py        # Replaces create_database.py + import_to_postgres.py
│   ├── analysis/              # Will replace EDA and analysis scripts
│   ├── models/                # Will replace modeling scripts
│   └── utils/
│       ├── config.py          # New: centralized configuration
│       └── helpers.py         # New: common utilities
├── scripts/
│   └── run_full_pipeline.py   # New: orchestration script
├── config/
│   └── config.yaml            # New: configuration file
└── main.py                    # New: command-line interface
```

## Migration Steps

### 1. Update Your Workflow

**Old way:**
```bash
python download_dataset.py
python create_database.py
python import_to_postgres.py
```

**New way:**
```bash
python main.py --full-setup
# or
python scripts/run_full_pipeline.py
```

### 2. Configuration Changes

**Old way:** Hardcoded values in each script
```python
db_name = "ai_job_db"
db_user = "your_username"
# ... scattered throughout scripts
```

**New way:** Centralized configuration
```yaml
# config/config.yaml
database:
  name: "ai_job_db"
  user: "your_username"
  # ... all settings in one place
```

### 3. Import Changes

**Old way:** Direct script execution
```bash
python analyze_cluster_features.py
```

**New way:** Import and use classes
```python
from src.analysis.clustering import ClusterAnalyzer
analyzer = ClusterAnalyzer()
analyzer.analyze_features()
```

## Benefits You'll See

1. **Easier Maintenance**: Changes to database settings only need to be made in one place
2. **Better Error Handling**: Consistent error handling across all modules
3. **Reusable Code**: Functions can be imported and reused
4. **Testing**: Easier to write unit tests for individual functions
5. **Documentation**: Better code organization makes it self-documenting

## What's Next

The following scripts still need to be refactored into the new structure:

### Analysis Scripts (to be moved to `src/analysis/`)
- `eda_check_cleaning.py` → `src/analysis/eda.py`
- `visualize_outliers.py` → `src/analysis/outliers.py`
- `remove_outliers.py` → `src/analysis/cleaning.py`
- `analyze_cluster_features.py` → `src/analysis/clustering.py`
- `analyze_entry_level_requirements.py` → `src/analysis/entry_level.py`

### Model Scripts (to be moved to `src/models/`)
- `build_regression_model.py` → `src/models/regression.py`
- `build_classification_model.py` → `src/models/classification.py`
- `build_clustering_model.py` → `src/models/clustering.py`
- `improve_classification_model.py` → `src/models/classification.py` (enhancement)

## Testing the New Structure

1. **Test the downloader:**
   ```bash
   python main.py --download
   ```

2. **Test database operations:**
   ```bash
   python main.py --create-db
   python main.py --import-db
   ```

3. **Test the full pipeline:**
   ```bash
   python scripts/run_full_pipeline.py
   ```

## Rollback Plan

If you need to rollback, the original scripts are still available in the root directory. You can continue using them while the refactoring is in progress.

## Questions?

If you have questions about the new structure or need help migrating specific functionality, please refer to the new README_REFACTORED.md file or ask for assistance. 