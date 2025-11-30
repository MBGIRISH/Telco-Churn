"""
Advanced PyTorch Deep Learning Model
Neural Network with BatchNorm, Dropout, and advanced training features
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


class ChurnDataset(Dataset):
    """PyTorch Dataset for churn prediction."""
    
    def __init__(self, X, y):
        """
        Initialize dataset.
        
        Args:
            X: Features (DataFrame or numpy array)
            y: Target (array)
        """
        if isinstance(X, pd.DataFrame):
            self.X = torch.FloatTensor(X.values)
        else:
            self.X = torch.FloatTensor(X)
        
        if len(y.shape) == 1:
            self.y = torch.FloatTensor(y.reshape(-1, 1))
        else:
            self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class AdvancedANNModel(nn.Module):
    """Advanced Artificial Neural Network for churn prediction."""
    
    def __init__(self, input_dim: int, hidden_layers: List[int] = None, 
                 dropout_rate: float = 0.3, use_batch_norm: bool = True):
        """
        Initialize advanced ANN model.
        
        Args:
            input_dim: Number of input features
            hidden_layers: List of hidden layer sizes (default: [128, 64, 32])
            dropout_rate: Dropout rate
            use_batch_norm: Whether to use batch normalization
        """
        super(AdvancedANNModel, self).__init__()
        
        if hidden_layers is None:
            hidden_layers = [128, 64, 32]
        
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        # Build layers
        layers = []
        prev_size = input_dim
        
        for i, layer_size in enumerate(hidden_layers):
            # Linear layer
            layers.append(nn.Linear(prev_size, layer_size))
            
            # Batch normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(layer_size))
            
            # Activation
            layers.append(nn.ReLU())
            
            # Dropout
            layers.append(nn.Dropout(dropout_rate))
            
            prev_size = layer_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


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


def train_pytorch_model(X_train, y_train, X_val, y_val, X_test, y_test,
                       hidden_layers: List[int] = None,
                       epochs: int = 100, batch_size: int = 32,
                       learning_rate: float = 0.001, dropout_rate: float = 0.3,
                       use_batch_norm: bool = True, patience: int = 15,
                       save_model: bool = True) -> Dict:
    """
    Train advanced PyTorch ANN model.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        X_test: Test features
        y_test: Test target
        hidden_layers: List of hidden layer sizes
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        dropout_rate: Dropout rate
        use_batch_norm: Whether to use batch normalization
        patience: Early stopping patience
        save_model: Whether to save the model
        
    Returns:
        Dictionary with model, history, and metrics
    """
    logger.info("Training advanced PyTorch ANN model...")
    
    if hidden_layers is None:
        hidden_layers = [128, 64, 32]
    
    # Create datasets
    train_dataset = ChurnDataset(X_train, y_train)
    val_dataset = ChurnDataset(X_val, y_val)
    test_dataset = ChurnDataset(X_test, y_test)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Build model
    input_dim = X_train.shape[1]
    model = AdvancedANNModel(
        input_dim=input_dim,
        hidden_layers=hidden_layers,
        dropout_rate=dropout_rate,
        use_batch_norm=use_batch_norm
    )
    
    logger.info(f"Model architecture:")
    logger.info(f"Input dimension: {input_dim}")
    logger.info(f"Hidden layers: {hidden_layers}")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6, verbose=True
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'train_auc': [],
        'val_auc': []
    }
    
    best_val_loss = float('inf')
    best_val_auc = 0.0
    patience_counter = 0
    best_model_state = None
    
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
        train_probs = []
        train_labels = []
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            train_total += y_batch.size(0)
            train_correct += (predicted == y_batch).sum().item()
            
            train_probs.extend(outputs.detach().cpu().numpy())
            train_labels.extend(y_batch.cpu().numpy())
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_probs = []
        val_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                val_total += y_batch.size(0)
                val_correct += (predicted == y_batch).sum().item()
                
                val_probs.extend(outputs.cpu().numpy())
                val_labels.extend(y_batch.cpu().numpy())
        
        # Calculate averages
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        # Calculate AUC
        train_auc = roc_auc_score(np.array(train_labels).ravel(), np.array(train_probs).ravel())
        val_auc = roc_auc_score(np.array(val_labels).ravel(), np.array(val_probs).ravel())
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_auc'].append(train_auc)
        history['val_auc'].append(val_auc)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping based on validation AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"Epoch [{epoch+1}/{epochs}] - "
                       f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, Train AUC: {train_auc:.4f} | "
                       f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")
        
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Evaluate on test set
    model.eval()
    test_probs = []
    test_preds = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            test_probs.extend(outputs.cpu().numpy())
            test_preds.extend((outputs > 0.5).float().cpu().numpy())
    
    test_probs = np.array(test_probs).ravel()
    test_preds = np.array(test_preds).ravel()
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, test_preds),
        'precision': precision_score(y_test, test_preds),
        'recall': recall_score(y_test, test_preds),
        'f1_score': f1_score(y_test, test_preds),
        'roc_auc': roc_auc_score(y_test, test_probs)
    }
    
    logger.info(f"\nPyTorch ANN Model Metrics:")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1-Score: {metrics['f1_score']:.4f}")
    logger.info(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    
    # Save model
    if save_model:
        project_root = get_project_root()
        models_dir = project_root / "models"
        ensure_dir(models_dir)
        
        model_path = models_dir / "best_pytorch_model.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'input_dim': input_dim,
            'hidden_layers': hidden_layers,
            'dropout_rate': dropout_rate,
            'use_batch_norm': use_batch_norm,
            'metrics': metrics,
            'history': history
        }, model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save history as JSON
        history_path = models_dir / "pytorch_training_history.json"
        history_for_json = {k: [float(v) for v in values] for k, values in history.items()}
        save_json(history_for_json, history_path)
    
    return {
        'model': model,
        'history': history,
        'metrics': metrics,
        'predictions': test_preds,
        'probabilities': test_probs
    }


def main():
    """Main function to run PyTorch model training."""
    # Load processed data
    X_train, X_val, X_test, y_train, y_val, y_test = load_processed_data()
    
    # Train PyTorch model
    results = train_pytorch_model(
        X_train, y_train, X_val, y_val, X_test, y_test,
        hidden_layers=[128, 64, 32],
        epochs=100,
        batch_size=32,
        learning_rate=0.001,
        dropout_rate=0.3,
        use_batch_norm=True,
        patience=15
    )
    
    logger.info("\nPyTorch model training completed successfully!")
    return results


if __name__ == "__main__":
    results = main()

