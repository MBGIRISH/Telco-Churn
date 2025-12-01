"""
Streamlit Deployment Module
Helper functions for Streamlit app
"""

import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_model_and_preprocessors(model_name: str = "xgboost"):
    """
    Load model, scaler, and label encoder.
    
    Args:
        model_name: Name of the model to load
        
    Returns:
        Tuple of (model, scaler, label_encoder, feature_names)
    """
    # Get project root - handle both local and Render paths
    current_file = Path(__file__)
    if current_file.parent.name == "src":
        project_root = current_file.parent.parent
    else:
        # Fallback: try to find project root
        project_root = Path.cwd()
        if not (project_root / "models").exists():
            project_root = project_root.parent
    
    models_dir = project_root / "models"
    
    # Try to load tuned model first
    model_path = models_dir / f"{model_name}_tuned.pkl"
    if not model_path.exists():
        model_path = models_dir / f"{model_name}.pkl"
    
    # Try alternative locations
    if not model_path.exists():
        # Try in current directory
        alt_path = Path("models") / f"{model_name}.pkl"
        if alt_path.exists():
            model_path = alt_path
            models_dir = Path("models")
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Please train models first by running: python src/model_training.py\n"
            f"Or ensure models are uploaded to the models/ directory."
        )
    
    scaler_path = models_dir / "scaler.pkl"
    label_encoder_path = models_dir / "label_encoder.pkl"
    
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    if not label_encoder_path.exists():
        raise FileNotFoundError(f"Label encoder not found: {label_encoder_path}")
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(label_encoder_path)
    
    # Get feature names from training data
    data_dir = project_root / "data"
    X_train_path = data_dir / "X_train.csv"
    if X_train_path.exists():
        X_train = pd.read_csv(X_train_path)
        feature_names = X_train.columns.tolist()
    else:
        feature_names = None
    
    logger.info(f"Loaded model: {model_path.name}")
    return model, scaler, label_encoder, feature_names


def preprocess_input(gender: str, SeniorCitizen: int, Partner: str, Dependents: str,
                    tenure: int, PhoneService: str, MultipleLines: str,
                    InternetService: str, OnlineSecurity: str, OnlineBackup: str,
                    DeviceProtection: str, TechSupport: str, StreamingTV: str,
                    StreamingMovies: str, Contract: str, PaperlessBilling: str,
                    PaymentMethod: str, MonthlyCharges: float, TotalCharges: float,
                    scaler, feature_names: list) -> pd.DataFrame:
    """
    Preprocess user input for prediction.
    
    Args:
        All customer features
        scaler: Fitted scaler
        feature_names: List of feature names
        
    Returns:
        Preprocessed DataFrame ready for prediction
    """
    # Create DataFrame from input
    input_data = {
        'gender': [gender],
        'SeniorCitizen': [SeniorCitizen],
        'Partner': [Partner],
        'Dependents': [Dependents],
        'tenure': [tenure],
        'PhoneService': [PhoneService],
        'MultipleLines': [MultipleLines],
        'InternetService': [InternetService],
        'OnlineSecurity': [OnlineSecurity],
        'OnlineBackup': [OnlineBackup],
        'DeviceProtection': [DeviceProtection],
        'TechSupport': [TechSupport],
        'StreamingTV': [StreamingTV],
        'StreamingMovies': [StreamingMovies],
        'Contract': [Contract],
        'PaperlessBilling': [PaperlessBilling],
        'PaymentMethod': [PaymentMethod],
        'MonthlyCharges': [MonthlyCharges],
        'TotalCharges': [TotalCharges]
    }
    
    df = pd.DataFrame(input_data)
    
    # Encode categorical features (same as in feature_engineering.py)
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from feature_engineering import encode_categorical_features
    
    # Handle case when target_col is None - add dummy Churn column for encoding
    if 'Churn' not in df.columns:
        df['Churn'] = 'No'  # Dummy value for encoding
    
    df_encoded, _ = encode_categorical_features(df, target_col='Churn')
    
    # Remove Churn column if it was added or exists
    if 'Churn' in df_encoded.columns:
        df_encoded = df_encoded.drop(columns=['Churn'])
    
    # Ensure all features from training are present
    if feature_names:
        for col in feature_names:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
    
        # Reorder columns to match training data
        df_encoded = df_encoded[feature_names]
    
    # Scale features
    X_scaled = pd.DataFrame(
        scaler.transform(df_encoded),
        columns=df_encoded.columns,
        index=df_encoded.index
    )
    
    return X_scaled


def predict_churn(model, X_scaled) -> tuple:
    """
    Make churn prediction.
    
    Args:
        model: Trained model
        X_scaled: Preprocessed features
        
    Returns:
        Tuple of (prediction, probability)
    """
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0][1]
    
    return prediction, probability


def get_feature_importance(model, feature_names: list, top_n: int = 10) -> pd.DataFrame:
    """
    Get feature importance from model.
    
    Args:
        model: Trained model
        feature_names: List of feature names
        top_n: Number of top features to return
        
    Returns:
        DataFrame with feature importance
    """
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)
        return importance_df
    elif hasattr(model, 'coef_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': np.abs(model.coef_[0])
        }).sort_values('importance', ascending=False).head(top_n)
        return importance_df
    else:
        return None

