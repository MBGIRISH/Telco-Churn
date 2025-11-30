"""
Deep Learning Model using PyTorch
Neural Network with 3-4 layers for churn prediction
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


class ChurnDataset(Dataset):
    """PyTorch Dataset for churn prediction."""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X.values if isinstance(X, pd.DataFrame) else X)
        self.y = torch.FloatTensor(y.reshape(-1, 1) if len(y.shape) == 1 else y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ANNModel(nn.Module):
    """Artificial Neural Network model for churn prediction."""
    
    def __init__(self, input_dim: int, layers_config: list = None, dropout_rate: float = 0.3):
        """
        Initialize ANN model.
        
        Args:
            input_dim: Number of input features
            layers_config: List of layer sizes (default: [32, 16])
            dropout_rate: Dropout rate
        """
        super(ANNModel, self).__init__()
        
        if layers_config is None:
            layers_config = [32, 16]
        
        self.layers_config = layers_config
        self.dropout_rate = dropout_rate
        
        # Build layers
        layers_list = []
        prev_size = input_dim
        
        for i, layer_size in enumerate(layers_config):
            layers_list.append(nn.Linear(prev_size, layer_size))
            layers_list.append(nn.BatchNorm1d(layer_size))
            layers_list.append(nn.ReLU())
            layers_list.append(nn.Dropout(dropout_rate))
            prev_size = layer_size
        
        # Output layer
        layers_list.append(nn.Linear(prev_size, 1))
        layers_list.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers_list)
    
    def forward(self, x):
        return self.model(x)


def load_processed_data():
    """Load processed training and test data."""
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    
    X_train = pd.read_csv(data_dir / "X_train.csv")
    X_test = pd.read_csv(data_dir / "X_test.csv")
    y_train = pd.read_csv(data_dir / "y_train.csv").values.ravel()
    y_test = pd.read_csv(data_dir / "y_test.csv").values.ravel()
    
    logger.info(f"Loaded data - Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def train_ann_model(X_train, y_train, X_test, y_test,
                   epochs: int = 50, batch_size: int = 32,
                   learning_rate: float = 0.001, validation_split: float = 0.2,
                   layers_config: list = None, save_model: bool = True) -> dict:
    """
    Train the PyTorch ANN model.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        validation_split: Validation split ratio
        layers_config: List of hidden layer sizes
        save_model: Whether to save the model
        
    Returns:
        Dictionary with model, history, and metrics
    """
    logger.info("Training PyTorch ANN model...")
    
    # Split training data for validation
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=validation_split, random_state=42, stratify=y_train
    )
    
    # Create datasets
    train_dataset = ChurnDataset(X_train_split, y_train_split)
    val_dataset = ChurnDataset(X_val_split, y_val_split)
    test_dataset = ChurnDataset(X_test, y_test)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Build model
    input_dim = X_train.shape[1]
    model = ANNModel(input_dim=input_dim, layers_config=layers_config)
    
    logger.info(f"Model architecture:")
    logger.info(f"Input dimension: {input_dim}")
    logger.info(f"Hidden layers: {layers_config if layers_config else [32, 16]}")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=0.0001
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10
    
    # Training loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    logger.info(f"Using device: {device}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            train_total += y_batch.size(0)
            train_correct += (predicted == y_batch).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                val_total += y_batch.size(0)
                val_correct += (predicted == y_batch).sum().item()
        
        # Calculate averages
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"Epoch [{epoch+1}/{epochs}] - "
                       f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                       f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            model.load_state_dict(best_model_state)
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Evaluate on test set
    model.eval()
    y_pred_proba_list = []
    y_pred_list = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            y_pred_proba_list.append(outputs.cpu().numpy())
            y_pred_list.append((outputs > 0.5).float().cpu().numpy())
    
    y_pred_proba = np.concatenate(y_pred_proba_list).ravel()
    y_pred = np.concatenate(y_pred_list).ravel()
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    logger.info(f"\nPyTorch ANN Model Metrics:")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1-Score: {metrics['f1_score']:.4f}")
    logger.info(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    
    # Save model
    if save_model:
        project_root = Path(__file__).parent.parent
        model_path = project_root / "models" / "ann_model_pytorch.pth"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'input_dim': input_dim,
            'layers_config': layers_config if layers_config else [32, 16],
            'metrics': metrics
        }, model_path)
        logger.info(f"Model saved to {model_path}")
    
    return {
        'model': model,
        'history': history,
        'metrics': metrics,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }


def compare_with_ml_models(ann_results: dict, ml_results: dict = None):
    """
    Compare ANN performance with traditional ML models.
    
    Args:
        ann_results: ANN model results
        ml_results: Dictionary with ML model results (optional)
    """
    logger.info("\n" + "="*60)
    logger.info("DEEP LEARNING vs MACHINE LEARNING COMPARISON")
    logger.info("="*60)
    
    comparison_data = {
        'Model': ['ANN (PyTorch)'],
        'Accuracy': [ann_results['metrics']['accuracy']],
        'Precision': [ann_results['metrics']['precision']],
        'Recall': [ann_results['metrics']['recall']],
        'F1-Score': [ann_results['metrics']['f1_score']],
        'ROC-AUC': [ann_results['metrics']['roc_auc']]
    }
    
    # Add ML models if provided
    if ml_results:
        for model_name, results in ml_results.items():
            if results and 'metrics' in results:
                comparison_data['Model'].append(model_name)
                comparison_data['Accuracy'].append(results['metrics']['accuracy'])
                comparison_data['Precision'].append(results['metrics']['precision'])
                comparison_data['Recall'].append(results['metrics']['recall'])
                comparison_data['F1-Score'].append(results['metrics']['f1_score'])
                comparison_data['ROC-AUC'].append(results['metrics']['roc_auc'])
    
    comparison_df = pd.DataFrame(comparison_data)
    logger.info("\n" + comparison_df.to_string(index=False))
    
    # Save comparison
    project_root = Path(__file__).parent.parent
    comparison_path = project_root / "models" / "dl_vs_ml_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    logger.info(f"\nComparison saved to {comparison_path}")
    
    return comparison_df


def main():
    """Main function to run deep learning pipeline."""
    # Load processed data
    X_train, X_test, y_train, y_test = load_processed_data()
    
    # Train ANN model with PyTorch
    ann_results = train_ann_model(
        X_train, y_train, X_test, y_test,
        epochs=50, batch_size=32, learning_rate=0.001,
        validation_split=0.2, layers_config=[32, 16]
    )
    
    # Compare with ML models (if available)
    try:
        from model_training import load_processed_data as load_ml_data
        
        X_train_ml, X_test_ml, y_train_ml, y_test_ml = load_ml_data()
        
        # Load saved ML models instead of retraining
        project_root = Path(__file__).parent.parent
        models_dir = project_root / "models"
        
        ml_results = {}
        
        # Load Logistic Regression
        try:
            lr_model = joblib.load(models_dir / "logistic_regression.pkl")
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            lr_pred = lr_model.predict(X_test_ml)
            lr_proba = lr_model.predict_proba(X_test_ml)[:, 1]
            ml_results['Logistic Regression'] = {
                'metrics': {
                    'accuracy': accuracy_score(y_test_ml, lr_pred),
                    'precision': precision_score(y_test_ml, lr_pred),
                    'recall': recall_score(y_test_ml, lr_pred),
                    'f1_score': f1_score(y_test_ml, lr_pred),
                    'roc_auc': roc_auc_score(y_test_ml, lr_proba)
                }
            }
        except Exception as e:
            logger.warning(f"Could not load Logistic Regression: {e}")
        
        # Load Random Forest
        try:
            rf_model = joblib.load(models_dir / "random_forest.pkl")
            rf_pred = rf_model.predict(X_test_ml)
            rf_proba = rf_model.predict_proba(X_test_ml)[:, 1]
            ml_results['Random Forest'] = {
                'metrics': {
                    'accuracy': accuracy_score(y_test_ml, rf_pred),
                    'precision': precision_score(y_test_ml, rf_pred),
                    'recall': recall_score(y_test_ml, rf_pred),
                    'f1_score': f1_score(y_test_ml, rf_pred),
                    'roc_auc': roc_auc_score(y_test_ml, rf_proba)
                }
            }
        except Exception as e:
            logger.warning(f"Could not load Random Forest: {e}")
        
        # Load XGBoost
        try:
            xgb_model = joblib.load(models_dir / "xgboost.pkl")
            xgb_pred = xgb_model.predict(X_test_ml)
            xgb_proba = xgb_model.predict_proba(X_test_ml)[:, 1]
            ml_results['XGBoost'] = {
                'metrics': {
                    'accuracy': accuracy_score(y_test_ml, xgb_pred),
                    'precision': precision_score(y_test_ml, xgb_pred),
                    'recall': recall_score(y_test_ml, xgb_pred),
                    'f1_score': f1_score(y_test_ml, xgb_pred),
                    'roc_auc': roc_auc_score(y_test_ml, xgb_proba)
                }
            }
        except Exception as e:
            logger.warning(f"Could not load XGBoost: {e}")
        
        comparison_df = compare_with_ml_models(ann_results, ml_results)
    except Exception as e:
        logger.warning(f"Could not compare with ML models: {e}")
        comparison_df = compare_with_ml_models(ann_results)
    
    logger.info("\nDeep learning training completed successfully!")
    return ann_results, comparison_df


if __name__ == "__main__":
    ann_results, comparison_df = main()
