"""
Model Comparison Utilities
Provides functions for comparing multiple ML models
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import joblib
import sys
import time
import io
from datetime import datetime

# Handle imports
try:
    from src.utils import get_project_root, ensure_dir
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.utils import get_project_root, ensure_dir

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_model_comparison_data() -> Optional[pd.DataFrame]:
    """Load model comparison CSV if it exists."""
    project_root = get_project_root()
    comparison_path = project_root / "models" / "model_comparison.csv"
    
    if comparison_path.exists():
        df = pd.read_csv(comparison_path)
        return df
    return None


def get_model_parameters(model_name: str) -> Dict:
    """Get parameters for a trained model."""
    project_root = get_project_root()
    models_dir = project_root / "models"
    model_path = models_dir / f"{model_name}.pkl"
    
    if not model_path.exists():
        return {}
    
    try:
        model = joblib.load(model_path)
        params = model.get_params() if hasattr(model, 'get_params') else {}
        return params
    except Exception as e:
        logger.warning(f"Could not load parameters for {model_name}: {e}")
        return {}


def measure_inference_time(model_name: str, n_samples: int = 100) -> float:
    """Measure inference time for a model."""
    project_root = get_project_root()
    models_dir = project_root / "models"
    data_dir = project_root / "data"
    
    model_path = models_dir / f"{model_name}.pkl"
    if not model_path.exists():
        return 0.0
    
    try:
        model = joblib.load(model_path)
        X_test = pd.read_csv(data_dir / "X_test.csv")
        X_sample = X_test.head(n_samples)
        
        # Warm up
        _ = model.predict(X_sample.head(1))
        
        # Measure inference time
        start_time = time.time()
        _ = model.predict(X_sample)
        end_time = time.time()
        
        return (end_time - start_time) / n_samples * 1000  # ms per sample
    except Exception as e:
        logger.warning(f"Could not measure inference time for {model_name}: {e}")
        return 0.0


def get_feature_importance_comparison(model_names: List[str]) -> pd.DataFrame:
    """Get feature importance comparison across models."""
    project_root = get_project_root()
    models_dir = project_root / "models"
    data_dir = project_root / "data"
    
    try:
        X_test = pd.read_csv(data_dir / "X_test.csv")
        feature_names = X_test.columns.tolist()
        
        importance_dict = {}
        for model_name in model_names:
            model_path = models_dir / f"{model_name}.pkl"
            if not model_path.exists():
                continue
            
            try:
                model = joblib.load(model_path)
                
                # Get feature importance
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    importances = np.abs(model.coef_[0])
                else:
                    continue
                
                importance_dict[model_name] = importances
            except Exception as e:
                logger.warning(f"Could not get importance for {model_name}: {e}")
                continue
        
        if importance_dict:
            importance_df = pd.DataFrame(importance_dict, index=feature_names)
            return importance_df
        
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error getting feature importance comparison: {e}")
        return pd.DataFrame()


def enhance_comparison_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Enhance comparison dataframe with additional metrics."""
    df_enhanced = df.copy()
    
    # Add inference time if not present
    if 'Inference_Time_ms' not in df_enhanced.columns:
        df_enhanced['Inference_Time_ms'] = 0.0
        for idx, row in df_enhanced.iterrows():
            model_name = row['Model'].lower().replace(' ', '_')
            df_enhanced.at[idx, 'Inference_Time_ms'] = measure_inference_time(model_name)
    
    # Add training time if not present (placeholder)
    if 'Training_Time_s' not in df_enhanced.columns:
        df_enhanced['Training_Time_s'] = 0.0
    
    # Calculate overall score
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    available_metrics = [m for m in metrics if m in df_enhanced.columns]
    
    if available_metrics:
        # Normalize and average
        for metric in available_metrics:
            max_val = df_enhanced[metric].max()
            if max_val > 0:
                df_enhanced[f'{metric}_normalized'] = df_enhanced[metric] / max_val
        
        norm_cols = [f'{m}_normalized' for m in available_metrics]
        df_enhanced['Overall_Score'] = df_enhanced[norm_cols].mean(axis=1)
    
    return df_enhanced


def export_comparison_to_csv(df: pd.DataFrame, filename: str = None) -> str:
    """Export comparison dataframe to CSV string."""
    if filename is None:
        filename = f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue()


def create_radar_chart_data(df: pd.DataFrame) -> Tuple[List[str], Dict]:
    """Prepare data for radar chart."""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    available_metrics = [m for m in metrics if m in df.columns]
    
    radar_data = {}
    for _, row in df.iterrows():
        model_name = row['Model']
        values = [row[m] for m in available_metrics]
        radar_data[model_name] = values
    
    return available_metrics, radar_data

