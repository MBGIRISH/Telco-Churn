"""
SHAP Utilities for Streamlit Dashboard
Provides all SHAP visualization and computation functions
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import joblib
import sys
import base64
import io

# Handle imports
try:
    from src.utils import get_project_root, ensure_dir
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.utils import get_project_root, ensure_dir

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available. Install with: pip install shap")


def load_model_and_data(model_name: str = "best_model"):
    """Load model and test data."""
    project_root = get_project_root()
    models_dir = project_root / "models"
    data_dir = project_root / "data"
    
    # Load model
    model_path = models_dir / f"{model_name}.pkl"
    if not model_path.exists():
        # Try alternative names
        for alt_name in ["random_forest", "xgboost", "lightgbm", "logistic_regression"]:
            alt_path = models_dir / f"{alt_name}.pkl"
            if alt_path.exists():
                model_path = alt_path
                break
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = joblib.load(model_path)
    
    # Load data
    X_test = pd.read_csv(data_dir / "X_test.csv")
    y_test = pd.read_csv(data_dir / "y_test.csv").values.ravel()
    
    # Get feature names
    feature_names = X_test.columns.tolist()
    
    return model, X_test, y_test, feature_names


def create_shap_explainer(model, X_sample: pd.DataFrame, model_type: str = "auto"):
    """Create SHAP explainer for the model."""
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP is not available. Install with: pip install shap")
    
    logger.info(f"Creating SHAP explainer (type: {model_type})...")
    
    # Auto-detect model type
    if model_type == "auto":
        model_class = type(model).__name__.lower()
        if 'tree' in model_class or 'forest' in model_class or 'gbm' in model_class or 'boost' in model_class:
            model_type = "tree"
        elif 'linear' in model_class or 'logistic' in model_class:
            model_type = "linear"
        else:
            model_type = "kernel"
    
    # Create appropriate explainer
    if model_type == "tree":
        model_class_name = type(model).__name__.lower()
        is_xgboost = 'xgboost' in model_class_name or 'xgb' in model_class_name
        is_lightgbm = 'lgbm' in model_class_name or 'lightgbm' in model_class_name
        
        try:
            if is_xgboost:
                X_sample_array = X_sample.values if isinstance(X_sample, pd.DataFrame) else X_sample
                X_sample_small = X_sample_array[:min(100, len(X_sample_array))]
                try:
                    booster = model.get_booster()
                    explainer = shap.TreeExplainer(booster, X_sample_small)
                except:
                    explainer = shap.TreeExplainer(model, X_sample_small)
            elif is_lightgbm:
                X_sample_array = X_sample.values if isinstance(X_sample, pd.DataFrame) else X_sample
                X_sample_small = X_sample_array[:min(100, len(X_sample_array))]
                explainer = shap.TreeExplainer(model, X_sample_small)
            else:
                explainer = shap.TreeExplainer(model)
        except Exception as e:
            logger.warning(f"TreeExplainer failed: {e}. Using Explainer API...")
            X_sample_array = X_sample.values if isinstance(X_sample, pd.DataFrame) else X_sample
            X_sample_small = X_sample_array[:min(100, len(X_sample_array))]
            explainer = shap.Explainer(model, X_sample_small)
    elif model_type == "linear":
        explainer = shap.LinearExplainer(model, X_sample)
    else:
        X_sample_array = X_sample.values if isinstance(X_sample, pd.DataFrame) else X_sample
        X_sample_small = X_sample_array[:min(100, len(X_sample_array))]
        explainer = shap.Explainer(model, X_sample_small)
    
    logger.info(f"SHAP explainer created: {type(explainer).__name__}")
    return explainer


def calculate_shap_values(explainer, X_sample: pd.DataFrame, max_samples: int = 100):
    """Calculate SHAP values."""
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP is not available")
    
    logger.info(f"Calculating SHAP values for {min(max_samples, len(X_sample))} samples...")
    
    # Sample data if too large
    if len(X_sample) > max_samples:
        X_sample = X_sample.sample(max_samples, random_state=42)
    
    # Convert DataFrame to numpy array if needed
    X_sample_array = X_sample.values if isinstance(X_sample, pd.DataFrame) else X_sample
    
    try:
        shap_values = explainer.shap_values(X_sample_array)
    except Exception as e:
        logger.warning(f"Error calculating SHAP values: {e}. Trying with DataFrame...")
        shap_values = explainer.shap_values(X_sample)
    
    # Handle multi-class output (take positive class)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Positive class
    
    logger.info(f"SHAP values calculated. Shape: {shap_values.shape}")
    return shap_values, X_sample


def get_shap_summary_data(shap_values: np.ndarray, X_sample: pd.DataFrame, 
                          feature_names: List[str]) -> pd.DataFrame:
    """Get SHAP summary data for visualization."""
    # Calculate mean absolute SHAP values
    importance = np.abs(shap_values).mean(axis=0)
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'shap_importance': importance,
        'mean_shap': shap_values.mean(axis=0),
        'std_shap': shap_values.std(axis=0)
    }).sort_values('shap_importance', ascending=False)
    
    return importance_df


def create_beeswarm_plot(shap_values: np.ndarray, X_sample: pd.DataFrame, 
                        feature_names: List[str]) -> Optional[object]:
    """Create SHAP beeswarm plot."""
    if not SHAP_AVAILABLE:
        return None
    
    try:
        X_sample_array = X_sample.values if isinstance(X_sample, pd.DataFrame) else X_sample
        
        explanation = shap.Explanation(
            values=shap_values,
            data=X_sample_array,
            feature_names=feature_names
        )
        
        return explanation
    except Exception as e:
        logger.error(f"Error creating beeswarm plot: {e}")
        return None


def create_waterfall_plot(shap_values: np.ndarray, X_sample: pd.DataFrame,
                         feature_names: List[str], sample_idx: int = 0,
                         base_value: float = 0.0) -> Optional[object]:
    """Create SHAP waterfall plot for a single sample."""
    if not SHAP_AVAILABLE:
        return None
    
    try:
        X_sample_array = X_sample.values if isinstance(X_sample, pd.DataFrame) else X_sample
        
        explanation = shap.Explanation(
            values=shap_values[sample_idx],
            base_values=base_value,
            data=X_sample_array[sample_idx],
            feature_names=feature_names
        )
        
        return explanation
    except Exception as e:
        logger.error(f"Error creating waterfall plot: {e}")
        return None


def create_force_plot_data(shap_values: np.ndarray, X_sample: pd.DataFrame,
                          feature_names: List[str], sample_idx: int = 0,
                          base_value: float = 0.0) -> Dict:
    """Create data for force plot visualization."""
    X_sample_array = X_sample.values if isinstance(X_sample, pd.DataFrame) else X_sample
    
    return {
        'shap_values': shap_values[sample_idx].tolist(),
        'feature_values': X_sample_array[sample_idx].tolist(),
        'feature_names': feature_names,
        'base_value': base_value,
        'prediction': float(base_value + shap_values[sample_idx].sum())
    }


def export_shap_to_csv(shap_values: np.ndarray, X_sample: pd.DataFrame,
                      feature_names: List[str]) -> str:
    """Export SHAP values to CSV format string."""
    # Create DataFrame with SHAP values
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    
    # Add feature values
    X_sample_array = X_sample.values if isinstance(X_sample, pd.DataFrame) else X_sample
    for i, feat in enumerate(feature_names):
        shap_df[f'{feat}_value'] = X_sample_array[:, i]
    
    # Add row index
    shap_df.insert(0, 'sample_index', range(len(shap_df)))
    
    # Convert to CSV string
    csv_buffer = io.StringIO()
    shap_df.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue()


def compute_shap_analysis(model_name: str = "best_model", max_samples: int = 100) -> Optional[Dict]:
    """Complete SHAP analysis pipeline."""
    if not SHAP_AVAILABLE:
        logger.error("SHAP is not available. Install with: pip install shap")
        return None
    
    logger.info("Starting SHAP explainability analysis...")
    
    # Load model and data
    model, X_test, y_test, feature_names = load_model_and_data(model_name)
    
    # Create explainer
    explainer = create_shap_explainer(model, X_test)
    
    # Calculate SHAP values
    shap_values, X_sample = calculate_shap_values(explainer, X_test, max_samples)
    
    # Get summary data
    summary_df = get_shap_summary_data(shap_values, X_sample, feature_names)
    
    # Calculate base value (mean prediction)
    try:
        X_sample_array = X_sample.values if isinstance(X_sample, pd.DataFrame) else X_sample
        base_value = float(model.predict_proba(X_sample_array)[:, 1].mean())
    except:
        base_value = 0.0
    
    return {
        'explainer': explainer,
        'shap_values': shap_values,
        'X_sample': X_sample,
        'feature_names': feature_names,
        'summary_df': summary_df,
        'base_value': base_value,
        'model': model
    }

