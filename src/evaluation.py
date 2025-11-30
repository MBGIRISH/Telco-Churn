"""
Comprehensive Model Evaluation Module
Generates confusion matrices, ROC curves, PR curves, and feature importance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)
import joblib
import logging
from pathlib import Path
from typing import Dict, Optional
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

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


def load_processed_data():
    """Load processed test data."""
    project_root = get_project_root()
    data_dir = project_root / "data"
    
    X_test = pd.read_csv(data_dir / "X_test.csv")
    y_test = pd.read_csv(data_dir / "y_test.csv").values.ravel()
    
    return X_test, y_test


def evaluate_model(model, X_test, y_test, model_name: str = "Model") -> Dict:
    """
    Evaluate a model with comprehensive metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        model_name: Name of the model
        
    Returns:
        Dictionary with metrics and predictions
    """
    logger.info(f"Evaluating {model_name}...")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'average_precision': average_precision_score(y_test, y_pred_proba)
    }
    
    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    logger.info(f"\n{model_name} Metrics:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.4f}")
    
    return {
        'metrics': metrics,
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'classification_report': report
    }


def plot_confusion_matrix(y_test, y_pred, model_name: str, save_path: Optional[Path] = None):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        ensure_dir(save_path.parent)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
    plt.close()


def plot_roc_curve(y_test, y_pred_proba, model_name: str, save_path: Optional[Path] = None):
    """Plot ROC curve."""
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        ensure_dir(save_path.parent)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"ROC curve saved to {save_path}")
    plt.close()


def plot_pr_curve(y_test, y_pred_proba, model_name: str, save_path: Optional[Path] = None):
    """Plot Precision-Recall curve."""
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkblue', lw=2,
             label=f'PR curve (AP = {avg_precision:.2f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'Precision-Recall Curve - {model_name}', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        ensure_dir(save_path.parent)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"PR curve saved to {save_path}")
    plt.close()


def plot_feature_importance(model, feature_names: list, model_name: str, 
                            top_n: int = 15, save_path: Optional[Path] = None):
    """Plot feature importance for tree-based models."""
    importance_df = None
    
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)
    elif hasattr(model, 'coef_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': np.abs(model.coef_[0])
        }).sort_values('importance', ascending=False).head(top_n)
    
    if importance_df is not None and len(importance_df) > 0:
        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
        plt.title(f'Top {top_n} Feature Importance - {model_name}', fontsize=14, fontweight='bold')
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            ensure_dir(save_path.parent)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        plt.close()
        
        return importance_df
    
    return None


def plot_training_curves(history: Dict, model_name: str, save_path: Optional[Path] = None):
    """Plot PyTorch training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Training and Validation Loss', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy curves
    axes[0, 1].plot(history['train_acc'], label='Train Accuracy')
    axes[0, 1].plot(history['val_acc'], label='Val Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # AUC curves
    if 'train_auc' in history and 'val_auc' in history:
        axes[1, 0].plot(history['train_auc'], label='Train AUC')
        axes[1, 0].plot(history['val_auc'], label='Val AUC')
        axes[1, 0].set_title('Training and Validation AUC', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        ensure_dir(save_path.parent)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training curves saved to {save_path}")
    plt.close()


def evaluate_all_models():
    """Evaluate all saved models."""
    project_root = get_project_root()
    models_dir = project_root / "models"
    output_dir = project_root / "notebooks" / "evaluation_plots"
    ensure_dir(output_dir)
    
    # Load test data
    X_test, y_test = load_processed_data()
    
    # Load feature names
    X_train_path = project_root / "data" / "X_train.csv"
    if X_train_path.exists():
        X_train = pd.read_csv(X_train_path)
        feature_names = X_train.columns.tolist()
    else:
        feature_names = X_test.columns.tolist()
    
    # Load models
    models_dict = {}
    model_files = {
        'Logistic Regression': 'logistic_regression.pkl',
        'Random Forest': 'random_forest.pkl',
        'LightGBM': 'lightgbm.pkl',
        'XGBoost': 'xgboost.pkl',
        'CatBoost': 'catboost.pkl'
    }
    
    for name, filename in model_files.items():
        model_path = models_dir / filename
        if model_path.exists():
            try:
                models_dict[name] = joblib.load(model_path)
                logger.info(f"Loaded {name}")
            except Exception as e:
                logger.warning(f"Could not load {name}: {e}")
    
    if not models_dict:
        logger.error("No models found to evaluate!")
        return None
    
    # Evaluate and create plots
    results = {}
    comparison_data = []
    
    for model_name, model in models_dict.items():
        result = evaluate_model(model, X_test, y_test, model_name)
        results[model_name] = result
        
        # Create plots
        plot_confusion_matrix(
            y_test, result['predictions'], model_name,
            save_path=output_dir / f"{model_name}_confusion_matrix.png"
        )
        
        plot_roc_curve(
            y_test, result['probabilities'], model_name,
            save_path=output_dir / f"{model_name}_roc_curve.png"
        )
        
        plot_pr_curve(
            y_test, result['probabilities'], model_name,
            save_path=output_dir / f"{model_name}_pr_curve.png"
        )
        
        # Feature importance
        importance_df = plot_feature_importance(
            model, feature_names, model_name,
            save_path=output_dir / f"{model_name}_feature_importance.png"
        )
        
        # Add to comparison
        comparison_data.append({
            'Model': model_name,
            'Accuracy': result['metrics']['accuracy'],
            'Precision': result['metrics']['precision'],
            'Recall': result['metrics']['recall'],
            'F1-Score': result['metrics']['f1_score'],
            'ROC-AUC': result['metrics']['roc_auc'],
            'Avg Precision': result['metrics']['average_precision']
        })
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('ROC-AUC', ascending=False)
    
    logger.info("\n" + "="*60)
    logger.info("MODEL COMPARISON SUMMARY")
    logger.info("="*60)
    logger.info("\n" + comparison_df.to_string(index=False))
    
    # Save comparison
    comparison_path = output_dir / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    logger.info(f"\nComparison saved to {comparison_path}")
    
    return results, comparison_df


def main():
    """Main function to run evaluation pipeline."""
    results, comparison_df = evaluate_all_models()
    
    if results:
        logger.info("\nModel evaluation completed successfully!")
        return results, comparison_df
    else:
        logger.error("Evaluation failed - no models found")
        return None, None


if __name__ == "__main__":
    results, comparison_df = main()
