"""
Utility functions for the Customer Churn Prediction System
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    if config_path is None:
        config_path = get_project_root() / "config.json"
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        logger.warning(f"Config file not found at {config_path}, using defaults")
        return {}


def save_config(config: Dict[str, Any], config_path: Optional[Path] = None):
    """Save configuration to JSON file."""
    if config_path is None:
        config_path = get_project_root() / "config.json"
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    logger.info(f"Config saved to {config_path}")


def ensure_dir(path: Path):
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)


def save_json(data: Dict[str, Any], filepath: Path):
    """Save dictionary to JSON file."""
    ensure_dir(filepath.parent)
    
    # Convert numpy types to Python native types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    data_serializable = convert_to_serializable(data)
    
    with open(filepath, 'w') as f:
        json.dump(data_serializable, f, indent=4)
    logger.info(f"Saved JSON to {filepath}")


def load_json(filepath: Path) -> Dict[str, Any]:
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def calculate_class_weights(y: np.ndarray) -> Dict[int, float]:
    """Calculate class weights for imbalanced datasets."""
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    return dict(zip(classes, weights))


def get_feature_names_from_dataframe(df: pd.DataFrame, exclude_cols: list = None) -> list:
    """Get feature names from dataframe, excluding specified columns."""
    if exclude_cols is None:
        exclude_cols = []
    return [col for col in df.columns if col not in exclude_cols]


def log_model_info(model, model_name: str):
    """Log model information."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Model: {model_name}")
    logger.info(f"{'='*60}")
    logger.info(f"Type: {type(model).__name__}")
    
    if hasattr(model, 'get_params'):
        params = model.get_params()
        logger.info(f"Parameters: {len(params)}")
        for key, value in list(params.items())[:5]:  # Show first 5
            logger.info(f"  {key}: {value}")
        if len(params) > 5:
            logger.info(f"  ... and {len(params) - 5} more")


def format_metrics(metrics: Dict[str, float]) -> str:
    """Format metrics dictionary for display."""
    formatted = []
    for key, value in metrics.items():
        if isinstance(value, float):
            formatted.append(f"{key}: {value:.4f}")
        else:
            formatted.append(f"{key}: {value}")
    return "\n".join(formatted)

