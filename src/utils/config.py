import yaml
import os
from pathlib import Path
from dotenv import load_dotenv

def load_config(config_path="config/config.yaml"):
    """Load configuration from YAML file with environment variable substitution."""
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load environment variables from .env file
    load_dotenv()
    
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    
    # Replace environment variables in the config
    config = _replace_env_vars(config)
    
    return config

def _replace_env_vars(obj):
    """Recursively replace environment variable placeholders in config."""
    if isinstance(obj, dict):
        return {key: _replace_env_vars(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_replace_env_vars(item) for item in obj]
    elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        env_var = obj[2:-1]  # Remove ${ and }
        value = os.getenv(env_var)
        if value is None:
            raise ValueError(f"Environment variable {env_var} not found")
        return value
    else:
        return obj

def get_config():
    """Get the configuration dictionary."""
    return load_config() 