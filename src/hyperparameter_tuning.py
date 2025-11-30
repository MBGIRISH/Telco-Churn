"""
Hyperparameter Tuning using Optuna
Optimizes LightGBM, XGBoost, and CatBoost hyperparameters
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Optional
from sklearn.metrics import roc_auc_score
import joblib
import sys

# Handle imports
try:
    from src.utils import get_project_root, ensure_dir, save_json
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.utils import get_project_root, ensure_dir, save_json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import Optuna
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.error("Optuna not available. Install with: pip install optuna")

# Try to import models
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


def load_processed_data():
    """Load processed training and validation data."""
    project_root = get_project_root()
    data_dir = project_root / "data"
    
    X_train = pd.read_csv(data_dir / "X_train.csv")
    X_val = pd.read_csv(data_dir / "X_val.csv")
    y_train = pd.read_csv(data_dir / "y_train.csv").values.ravel()
    y_val = pd.read_csv(data_dir / "y_val.csv").values.ravel()
    
    logger.info(f"Loaded data - Train: {X_train.shape}, Val: {X_val.shape}")
    return X_train, X_val, y_train, y_val


def tune_lightgbm(X_train, y_train, X_val, y_val, n_trials: int = 50) -> Dict:
    """Tune LightGBM hyperparameters using Optuna."""
    if not OPTUNA_AVAILABLE or not LIGHTGBM_AVAILABLE:
        logger.warning("Optuna or LightGBM not available, skipping...")
        return None
    
    logger.info("Tuning LightGBM hyperparameters with Optuna...")
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 300),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': 42,
            'verbose': -1
        }
        
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                 eval_metric='auc', callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(0)])
        
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, y_pred_proba)
        return score
    
    study = optuna.create_study(direction='maximize', study_name='lightgbm_tuning')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    logger.info(f"Best LightGBM parameters: {study.best_params}")
    logger.info(f"Best LightGBM ROC-AUC: {study.best_value:.4f}")
    
    # Train best model
    best_model = lgb.LGBMClassifier(**study.best_params, random_state=42, verbose=-1)
    best_model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                   eval_metric='auc', callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(0)])
    
    # Save best parameters
    project_root = get_project_root()
    params_path = project_root / "models" / "lightgbm_best_params.json"
    save_json(study.best_params, params_path)
    
    # Save best model
    model_path = project_root / "models" / "lightgbm_tuned.pkl"
    ensure_dir(model_path.parent)
    joblib.dump(best_model, model_path)
    logger.info(f"Tuned LightGBM model saved to {model_path}")
    
    return {
        'best_params': study.best_params,
        'best_score': study.best_value,
        'model': best_model,
        'study': study
    }


def tune_xgboost(X_train, y_train, X_val, y_val, n_trials: int = 50) -> Dict:
    """Tune XGBoost hyperparameters using Optuna."""
    if not OPTUNA_AVAILABLE or not XGBOOST_AVAILABLE:
        logger.warning("Optuna or XGBoost not available, skipping...")
        return None
    
    logger.info("Tuning XGBoost hyperparameters with Optuna...")
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': 42,
            'eval_metric': 'logloss',
            'use_label_encoder': False
        }
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, y_pred_proba)
        return score
    
    study = optuna.create_study(direction='maximize', study_name='xgboost_tuning')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    logger.info(f"Best XGBoost parameters: {study.best_params}")
    logger.info(f"Best XGBoost ROC-AUC: {study.best_value:.4f}")
    
    # Train best model
    best_model = xgb.XGBClassifier(**study.best_params, random_state=42, 
                                   eval_metric='logloss', use_label_encoder=False)
    best_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    # Save best parameters
    project_root = get_project_root()
    params_path = project_root / "models" / "xgboost_best_params.json"
    save_json(study.best_params, params_path)
    
    # Save best model
    model_path = project_root / "models" / "xgboost_tuned.pkl"
    ensure_dir(model_path.parent)
    joblib.dump(best_model, model_path)
    logger.info(f"Tuned XGBoost model saved to {model_path}")
    
    return {
        'best_params': study.best_params,
        'best_score': study.best_value,
        'model': best_model,
        'study': study
    }


def tune_catboost(X_train, y_train, X_val, y_val, n_trials: int = 50) -> Dict:
    """Tune CatBoost hyperparameters using Optuna."""
    if not OPTUNA_AVAILABLE or not CATBOOST_AVAILABLE:
        logger.warning("Optuna or CatBoost not available, skipping...")
        return None
    
    logger.info("Tuning CatBoost hyperparameters with Optuna...")
    
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 50, 300),
            'depth': trial.suggest_int('depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
            'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
            'random_state': 42,
            'verbose': False
        }
        
        model = cb.CatBoostClassifier(**params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=10)
        
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, y_pred_proba)
        return score
    
    study = optuna.create_study(direction='maximize', study_name='catboost_tuning')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    logger.info(f"Best CatBoost parameters: {study.best_params}")
    logger.info(f"Best CatBoost ROC-AUC: {study.best_value:.4f}")
    
    # Train best model
    best_model = cb.CatBoostClassifier(**study.best_params, random_state=42, verbose=False)
    best_model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=10)
    
    # Save best parameters
    project_root = get_project_root()
    params_path = project_root / "models" / "catboost_best_params.json"
    save_json(study.best_params, params_path)
    
    # Save best model
    model_path = project_root / "models" / "catboost_tuned.pkl"
    ensure_dir(model_path.parent)
    joblib.dump(best_model, model_path)
    logger.info(f"Tuned CatBoost model saved to {model_path}")
    
    return {
        'best_params': study.best_params,
        'best_score': study.best_value,
        'model': best_model,
        'study': study
    }


def tune_all_models(X_train, y_train, X_val, y_val, n_trials: int = 50) -> Dict:
    """Tune all available models."""
    if not OPTUNA_AVAILABLE:
        logger.error("Optuna is not available. Install with: pip install optuna")
        return {}
    
    logger.info("Starting hyperparameter tuning for all models...")
    results = {}
    
    if LIGHTGBM_AVAILABLE:
        results['lightgbm'] = tune_lightgbm(X_train, y_train, X_val, y_val, n_trials)
    
    if XGBOOST_AVAILABLE:
        results['xgboost'] = tune_xgboost(X_train, y_train, X_val, y_val, n_trials)
    
    if CATBOOST_AVAILABLE:
        results['catboost'] = tune_catboost(X_train, y_train, X_val, y_val, n_trials)
    
    logger.info("\nHyperparameter tuning completed!")
    return results


def main():
    """Main function to run hyperparameter tuning."""
    if not OPTUNA_AVAILABLE:
        logger.error("Please install Optuna: pip install optuna")
        return
    
    # Load processed data
    X_train, X_val, y_train, y_val = load_processed_data()
    
    # Tune all models
    results = tune_all_models(X_train, y_train, X_val, y_val, n_trials=50)
    
    return results


if __name__ == "__main__":
    results = main()
