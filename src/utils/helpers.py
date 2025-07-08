import os
import pandas as pd
from pathlib import Path

def ensure_directory(directory_path):
    """Ensure a directory exists, create if it doesn't."""
    Path(directory_path).mkdir(parents=True, exist_ok=True)

def save_plot(fig, filename, output_dir="outputs/plots"):
    """Save a matplotlib figure to the specified output directory."""
    ensure_directory(output_dir)
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {filepath}")

def save_model(model, filename, output_dir="outputs/models"):
    """Save a trained model to the specified output directory."""
    import joblib
    ensure_directory(output_dir)
    filepath = os.path.join(output_dir, filename)
    joblib.dump(model, filepath)
    print(f"Model saved to: {filepath}")

def load_model(filename, output_dir="outputs/models"):
    """Load a trained model from the specified output directory."""
    import joblib
    filepath = os.path.join(output_dir, filename)
    return joblib.load(filepath)

def save_report(content, filename, output_dir="outputs/reports"):
    """Save a text report to the specified output directory."""
    ensure_directory(output_dir)
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"Report saved to: {filepath}") 