"""
SHAP Explainability Module
Provides model interpretability using SHAP values
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Dict, List
import joblib
import sys

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
        for alt_name in ["random_forest", "xgboost", "lightgbm"]:
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
    """
    Create SHAP explainer for the model.
    
    Args:
        model: Trained model
        X_sample: Sample data for explainer
        model_type: Type of explainer ('tree', 'linear', 'kernel', 'auto')
        
    Returns:
        SHAP explainer
    """
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
        # Check if it's XGBoost or LightGBM (they need special handling)
        model_class_name = type(model).__name__.lower()
        is_xgboost = 'xgboost' in model_class_name or 'xgb' in model_class_name
        is_lightgbm = 'lgbm' in model_class_name or 'lightgbm' in model_class_name
        
        try:
            if is_xgboost:
                # For XGBoost, use the underlying booster directly
                X_sample_array = X_sample.values if isinstance(X_sample, pd.DataFrame) else X_sample
                X_sample_small = X_sample_array[:min(100, len(X_sample_array))]
                try:
                    # Try using the booster directly (avoids feature_names_in_ issue)
                    booster = model.get_booster()
                    explainer = shap.TreeExplainer(booster, X_sample_small)
                except:
                    # If that fails, use the model with data
                    explainer = shap.TreeExplainer(model, X_sample_small)
            elif is_lightgbm:
                # For LightGBM, convert DataFrame to numpy and use TreeExplainer
                X_sample_array = X_sample.values if isinstance(X_sample, pd.DataFrame) else X_sample
                X_sample_small = X_sample_array[:min(100, len(X_sample_array))]
                explainer = shap.TreeExplainer(model, X_sample_small)
            else:
                # For scikit-learn tree models, TreeExplainer works directly
                explainer = shap.TreeExplainer(model)
        except (AttributeError, TypeError, ValueError) as e:
            # If TreeExplainer fails, try with data sample
            logger.warning(f"TreeExplainer failed: {e}. Trying with data sample...")
            try:
                X_sample_array = X_sample.values if isinstance(X_sample, pd.DataFrame) else X_sample
                X_sample_small = X_sample_array[:min(100, len(X_sample_array))]
                explainer = shap.TreeExplainer(model, X_sample_small)
            except Exception as e2:
                logger.warning(f"TreeExplainer with data failed: {e2}. Using Explainer API...")
                # Use the newer Explainer API which handles XGBoost better
                X_sample_array = X_sample.values if isinstance(X_sample, pd.DataFrame) else X_sample
                X_sample_small = X_sample_array[:min(100, len(X_sample_array))]
                try:
                    explainer = shap.Explainer(model, X_sample_small)
                except Exception as e3:
                    logger.error(f"All SHAP explainer methods failed: {e3}")
                    raise RuntimeError(f"Could not create SHAP explainer for {type(model).__name__}. Error: {e3}")
    elif model_type == "linear":
        explainer = shap.LinearExplainer(model, X_sample)
    else:
        # Use Explainer API for other models (handles XGBoost and others better)
        X_sample_array = X_sample.values if isinstance(X_sample, pd.DataFrame) else X_sample
        X_sample_small = X_sample_array[:min(100, len(X_sample_array))]
        try:
            explainer = shap.Explainer(model, X_sample_small)
        except Exception as e:
            logger.warning(f"Explainer API failed: {e}. Using wrapped predict function...")
            # Last resort: wrap the predict function
            def wrapped_predict(X):
                if isinstance(X, pd.DataFrame):
                    X = X.values
                return model.predict_proba(X)
            explainer = shap.Explainer(wrapped_predict, X_sample_small)
    
    logger.info(f"SHAP explainer created: {type(explainer).__name__}")
    return explainer


def calculate_shap_values(explainer, X_sample: pd.DataFrame, max_samples: int = 100):
    """
    Calculate SHAP values.
    
    Args:
        explainer: SHAP explainer
        X_sample: Data to explain
        max_samples: Maximum number of samples to explain
        
    Returns:
        SHAP values
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP is not available")
    
    logger.info(f"Calculating SHAP values for {min(max_samples, len(X_sample))} samples...")
    
    # Sample data if too large
    if len(X_sample) > max_samples:
        X_sample = X_sample.sample(max_samples, random_state=42)
    
    # Convert DataFrame to numpy array if needed (for XGBoost compatibility)
    X_sample_array = X_sample.values if isinstance(X_sample, pd.DataFrame) else X_sample
    
    try:
        shap_values = explainer.shap_values(X_sample_array)
    except Exception as e:
        logger.warning(f"Error calculating SHAP values: {e}. Trying with DataFrame...")
        # Try with DataFrame if array fails
        shap_values = explainer.shap_values(X_sample)
    
    # Handle multi-class output (take positive class)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Positive class
    
    logger.info(f"SHAP values calculated. Shape: {shap_values.shape}")
    return shap_values, X_sample


def plot_shap_summary(shap_values, X_sample: pd.DataFrame, feature_names: List[str],
                     save_path: Optional[Path] = None):
    """Plot SHAP summary plot."""
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP is not available")
    
    logger.info("Creating SHAP summary plot...")
    
    # Convert DataFrame to numpy array if needed
    X_sample_array = X_sample.values if isinstance(X_sample, pd.DataFrame) else X_sample
    
    try:
        plt = shap.summary_plot(shap_values, X_sample_array, feature_names=feature_names, show=False)
    except Exception as e:
        logger.warning(f"Error with summary plot: {e}. Trying without feature names...")
        plt = shap.summary_plot(shap_values, X_sample_array, show=False)
    
    if save_path:
        ensure_dir(save_path.parent)
        import matplotlib.pyplot as plt_matplotlib
        plt_matplotlib.savefig(save_path, dpi=300, bbox_inches='tight')
        plt_matplotlib.close()
        logger.info(f"SHAP summary plot saved to {save_path}")


def plot_shap_beeswarm(shap_values, X_sample: pd.DataFrame, feature_names: List[str],
                      save_path: Optional[Path] = None):
    """Plot SHAP beeswarm plot."""
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP is not available")
    
    logger.info("Creating SHAP beeswarm plot...")
    
    # Convert DataFrame to numpy array if needed
    X_sample_array = X_sample.values if isinstance(X_sample, pd.DataFrame) else X_sample
    
    try:
        explanation = shap.Explanation(
            values=shap_values, 
            data=X_sample_array, 
            feature_names=feature_names
        )
        plt = shap.plots.beeswarm(explanation, show=False)
    except Exception as e:
        logger.warning(f"Error with beeswarm plot: {e}. Trying without feature names...")
        explanation = shap.Explanation(values=shap_values, data=X_sample_array)
        plt = shap.plots.beeswarm(explanation, show=False)
    
    if save_path:
        ensure_dir(save_path.parent)
        import matplotlib.pyplot as plt_matplotlib
        plt_matplotlib.savefig(save_path, dpi=300, bbox_inches='tight')
        plt_matplotlib.close()
        logger.info(f"SHAP beeswarm plot saved to {save_path}")


def plot_shap_waterfall(shap_values, X_sample: pd.DataFrame, feature_names: List[str],
                       sample_idx: int = 0, save_path: Optional[Path] = None):
    """Plot SHAP waterfall plot for a single sample."""
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP is not available")
    
    logger.info(f"Creating SHAP waterfall plot for sample {sample_idx}...")
    
    explanation = shap.Explanation(
        values=shap_values[sample_idx],
        base_values=shap_values[sample_idx].sum(),  # Approximate base value
        data=X_sample.iloc[sample_idx].values,
        feature_names=feature_names
    )
    
    plt = shap.plots.waterfall(explanation, show=False)
    
    if save_path:
        ensure_dir(save_path.parent)
        import matplotlib.pyplot as plt_matplotlib
        plt_matplotlib.savefig(save_path, dpi=300, bbox_inches='tight')
        plt_matplotlib.close()
        logger.info(f"SHAP waterfall plot saved to {save_path}")


def get_shap_feature_importance(shap_values: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
    """
    Get feature importance from SHAP values.
    
    Args:
        shap_values: SHAP values array
        feature_names: List of feature names
        
    Returns:
        DataFrame with feature importance
    """
    # Calculate mean absolute SHAP values
    importance = np.abs(shap_values).mean(axis=0)
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'shap_importance': importance
    }).sort_values('shap_importance', ascending=False)
    
    return importance_df


def explain_model(model_name: str = "best_model", max_samples: int = 100,
                 model_type: str = "auto") -> Dict:
    """
    Complete SHAP explainability pipeline.
    
    Args:
        model_name: Name of the model to explain
        max_samples: Maximum number of samples to explain
        model_type: Type of SHAP explainer
        
    Returns:
        Dictionary with explainer, SHAP values, and plots
    """
    if not SHAP_AVAILABLE:
        logger.error("SHAP is not available. Install with: pip install shap")
        return None
    
    logger.info("Starting SHAP explainability analysis...")
    
    # Load model and data
    model, X_test, y_test, feature_names = load_model_and_data(model_name)
    
    # Create explainer
    explainer = create_shap_explainer(model, X_test, model_type)
    
    # Calculate SHAP values
    shap_values, X_sample = calculate_shap_values(explainer, X_test, max_samples)
    
    # Create plots
    project_root = get_project_root()
    output_dir = project_root / "notebooks" / "shap_plots"
    ensure_dir(output_dir)
    
    plot_shap_summary(shap_values, X_sample, feature_names,
                     save_path=output_dir / "shap_summary.png")
    
    plot_shap_beeswarm(shap_values, X_sample, feature_names,
                      save_path=output_dir / "shap_beeswarm.png")
    
    plot_shap_waterfall(shap_values, X_sample, feature_names, sample_idx=0,
                       save_path=output_dir / "shap_waterfall_sample0.png")
    
    # Get feature importance
    importance_df = get_shap_feature_importance(shap_values, feature_names)
    importance_path = output_dir / "shap_feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    logger.info(f"SHAP feature importance saved to {importance_path}")
    
    logger.info("\nSHAP explainability analysis completed!")
    
    return {
        'explainer': explainer,
        'shap_values': shap_values,
        'X_sample': X_sample,
        'feature_names': feature_names,
        'importance_df': importance_df
    }


def explain_prediction(model, X_sample: pd.DataFrame, sample_idx: int,
                      explainer, shap_values: np.ndarray, feature_names: List[str]) -> Dict:
    """
    Explain a single prediction.
    
    Args:
        model: Trained model
        X_sample: Sample data
        sample_idx: Index of sample to explain
        explainer: SHAP explainer
        shap_values: Pre-calculated SHAP values
        feature_names: List of feature names
        
    Returns:
        Dictionary with prediction explanation
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP is not available")
    
    prediction = model.predict_proba(X_sample.iloc[[sample_idx]])[0]
    shap_value = shap_values[sample_idx]
    
    # Get top contributing features
    top_features = pd.DataFrame({
        'feature': feature_names,
        'shap_value': shap_value,
        'feature_value': X_sample.iloc[sample_idx].values
    }).sort_values('shap_value', key=abs, ascending=False)
    
    return {
        'prediction': prediction,
        'shap_values': shap_value,
        'top_features': top_features,
        'sample_data': X_sample.iloc[sample_idx]
    }


def main():
    """Main function to run SHAP explainability."""
    if not SHAP_AVAILABLE:
        logger.error("Please install SHAP: pip install shap")
        return None
    
    results = explain_model(model_name="best_model", max_samples=100)
    return results


if __name__ == "__main__":
    results = main()

