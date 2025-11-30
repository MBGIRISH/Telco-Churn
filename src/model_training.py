"""
Advanced Model Training Module
Trains LightGBM, XGBoost, and CatBoost models
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import logging
from pathlib import Path
from typing import Dict, Optional
import sys

# Handle imports
try:
    from src.utils import get_project_root, ensure_dir, log_model_info, format_metrics
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.utils import get_project_root, ensure_dir, log_model_info, format_metrics

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import advanced models
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not available")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    logger.warning("CatBoost not available")

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def load_processed_data():
    """Load processed training, validation, and test data."""
    project_root = get_project_root()
    data_dir = project_root / "data"
    
    X_train = pd.read_csv(data_dir / "X_train.csv")
    X_val = pd.read_csv(data_dir / "X_val.csv")
    X_test = pd.read_csv(data_dir / "X_test.csv")
    y_train = pd.read_csv(data_dir / "y_train.csv").values.ravel()
    y_val = pd.read_csv(data_dir / "y_val.csv").values.ravel()
    y_test = pd.read_csv(data_dir / "y_test.csv").values.ravel()
    
    logger.info(f"Loaded data - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate_model(model, X_test, y_test, model_name: str = "Model") -> Dict:
    """Evaluate model with comprehensive metrics."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    logger.info(f"\n{model_name} Metrics:")
    logger.info(format_metrics(metrics))
    
    return {'metrics': metrics, 'predictions': y_pred, 'probabilities': y_pred_proba}


def train_lightgbm(X_train, y_train, X_val, y_val, X_test, y_test,
                   save_model: bool = True) -> Dict:
    """Train LightGBM model."""
    if not LIGHTGBM_AVAILABLE:
        logger.warning("LightGBM not available, skipping...")
        return None
    
    logger.info("Training LightGBM model...")
    
    model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=7,
        learning_rate=0.1,
        random_state=42,
        verbose=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(0)]
    )
    
    log_model_info(model, "LightGBM")
    results = evaluate_model(model, X_test, y_test, "LightGBM")
    results['model'] = model
    
    if save_model:
        project_root = get_project_root()
        model_path = project_root / "models" / "lightgbm.pkl"
        ensure_dir(model_path.parent)
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
    
    return results


def train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test,
                  save_model: bool = True) -> Dict:
    """Train XGBoost model."""
    if not XGBOOST_AVAILABLE:
        logger.warning("XGBoost not available, skipping...")
        return None
    
    logger.info("Training XGBoost model...")
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    log_model_info(model, "XGBoost")
    results = evaluate_model(model, X_test, y_test, "XGBoost")
    results['model'] = model
    
    if save_model:
        project_root = get_project_root()
        model_path = project_root / "models" / "xgboost.pkl"
        ensure_dir(model_path.parent)
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
    
    return results


def train_catboost(X_train, y_train, X_val, y_val, X_test, y_test,
                   save_model: bool = True) -> Dict:
    """Train CatBoost model."""
    if not CATBOOST_AVAILABLE:
        logger.warning("CatBoost not available, skipping...")
        return None
    
    logger.info("Training CatBoost model...")
    
    model = cb.CatBoostClassifier(
        iterations=100,
        depth=6,
        learning_rate=0.1,
        random_state=42,
        verbose=False
    )
    
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=10
    )
    
    log_model_info(model, "CatBoost")
    results = evaluate_model(model, X_test, y_test, "CatBoost")
    results['model'] = model
    
    if save_model:
        project_root = get_project_root()
        model_path = project_root / "models" / "catboost.pkl"
        ensure_dir(model_path.parent)
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
    
    return results


def train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test,
                        save_model: bool = True) -> Dict:
    """Train Random Forest model."""
    logger.info("Training Random Forest model...")
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    log_model_info(model, "Random Forest")
    results = evaluate_model(model, X_test, y_test, "Random Forest")
    results['model'] = model
    
    if save_model:
        project_root = get_project_root()
        model_path = project_root / "models" / "random_forest.pkl"
        ensure_dir(model_path.parent)
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
    
    return results


def train_logistic_regression(X_train, y_train, X_val, y_val, X_test, y_test,
                              save_model: bool = True) -> Dict:
    """Train Logistic Regression model."""
    logger.info("Training Logistic Regression model...")
    
    model = LogisticRegression(random_state=42, max_iter=1000, solver='lbfgs')
    model.fit(X_train, y_train)
    
    log_model_info(model, "Logistic Regression")
    results = evaluate_model(model, X_test, y_test, "Logistic Regression")
    results['model'] = model
    
    if save_model:
        project_root = get_project_root()
        model_path = project_root / "models" / "logistic_regression.pkl"
        ensure_dir(model_path.parent)
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
    
    return results


def train_all_models(X_train, y_train, X_val, y_val, X_test, y_test) -> Dict:
    """Train all models and return results."""
    logger.info("Training all advanced models...")
    
    results = {}
    
    # Train all available models
    results['logistic_regression'] = train_logistic_regression(X_train, y_train, X_val, y_val, X_test, y_test)
    results['random_forest'] = train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test)
    
    if LIGHTGBM_AVAILABLE:
        results['lightgbm'] = train_lightgbm(X_train, y_train, X_val, y_val, X_test, y_test)
    
    if XGBOOST_AVAILABLE:
        results['xgboost'] = train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test)
    
    if CATBOOST_AVAILABLE:
        results['catboost'] = train_catboost(X_train, y_train, X_val, y_val, X_test, y_test)
    
    # Create comparison DataFrame
    comparison_data = []
    for model_name, result in results.items():
        if result and 'metrics' in result:
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Accuracy': result['metrics']['accuracy'],
                'Precision': result['metrics']['precision'],
                'Recall': result['metrics']['recall'],
                'F1-Score': result['metrics']['f1_score'],
                'ROC-AUC': result['metrics']['roc_auc']
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('ROC-AUC', ascending=False)
    
    logger.info("\n" + "="*60)
    logger.info("MODEL COMPARISON")
    logger.info("="*60)
    logger.info("\n" + comparison_df.to_string(index=False))
    
    # Save comparison
    project_root = get_project_root()
    comparison_path = project_root / "models" / "model_comparison.csv"
    ensure_dir(comparison_path.parent)
    comparison_df.to_csv(comparison_path, index=False)
    logger.info(f"\nComparison saved to {comparison_path}")
    
    # Find best model
    best_model_name = comparison_df.loc[comparison_df['ROC-AUC'].idxmax(), 'Model']
    logger.info(f"\nBest model based on ROC-AUC: {best_model_name}")
    
    # Save best model
    best_model_key = best_model_name.lower().replace(' ', '_')
    if best_model_key in results and results[best_model_key]:
        best_model = results[best_model_key]['model']
        best_model_path = project_root / "models" / "best_model.pkl"
        joblib.dump(best_model, best_model_path)
        logger.info(f"Best model saved to {best_model_path}")
    
    return results


def main():
    """Main function to run model training pipeline."""
    # Load processed data
    X_train, X_val, X_test, y_train, y_val, y_test = load_processed_data()
    
    # Train all models
    results = train_all_models(X_train, y_train, X_val, y_val, X_test, y_test)
    
    logger.info("\nModel training completed successfully!")
    return results


if __name__ == "__main__":
    results = main()
