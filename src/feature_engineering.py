"""
Advanced Feature Engineering Module
Includes CLV calculation, interaction features, and tenure binning
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import joblib
import logging
from pathlib import Path
from typing import Tuple, Optional
import sys

# Handle imports for both direct execution and module import
try:
    from src.utils import get_project_root, ensure_dir
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.utils import get_project_root, ensure_dir

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_cleaned_data(file_path: Optional[Path] = None) -> pd.DataFrame:
    """Load cleaned data."""
    if file_path is None:
        project_root = get_project_root()
        file_path = project_root / "data" / "cleaned_data.csv"
    
    logger.info(f"Loading cleaned data from {file_path}")
    df = pd.read_csv(file_path)
    return df


def create_tenure_bins(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create tenure bins for better feature representation.
    
    Args:
        df: DataFrame with tenure column
        
    Returns:
        DataFrame with tenure_bin column
    """
    df = df.copy()
    
    # Create tenure bins
    bins = [0, 12, 24, 36, 48, 60, 72, float('inf')]
    labels = ['0-12', '13-24', '25-36', '37-48', '49-60', '61-72', '73+']
    df['tenure_bin'] = pd.cut(df['tenure'], bins=bins, labels=labels, right=False)
    
    logger.info("Created tenure bins")
    return df


def calculate_clv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Customer Lifetime Value (CLV).
    
    CLV = (Average Monthly Charges * Average Tenure) - Total Charges
    
    Args:
        df: DataFrame with MonthlyCharges, tenure, TotalCharges
        
    Returns:
        DataFrame with CLV column
    """
    df = df.copy()
    
    # Calculate CLV
    df['CLV'] = (df['MonthlyCharges'] * df['tenure']) - df['TotalCharges']
    
    # Alternative: Projected CLV based on contract type
    contract_multiplier = {
        'Month-to-month': 1.0,
        'One year': 12.0,
        'Two year': 24.0
    }
    
    df['projected_CLV'] = df['MonthlyCharges'] * df['Contract'].map(contract_multiplier)
    df['CLV_ratio'] = df['TotalCharges'] / (df['projected_CLV'] + 1e-6)  # Avoid division by zero
    
    logger.info("Calculated Customer Lifetime Value (CLV)")
    return df


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features.
    
    Args:
        df: DataFrame
        
    Returns:
        DataFrame with interaction features
    """
    df = df.copy()
    
    # Monthly charges per tenure month
    df['charges_per_month'] = df['TotalCharges'] / (df['tenure'] + 1e-6)
    
    # Average monthly charge ratio
    df['avg_monthly_ratio'] = df['TotalCharges'] / (df['MonthlyCharges'] * df['tenure'] + 1e-6)
    
    # Service count (count of Yes services)
    service_cols = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['service_count'] = df[service_cols].apply(
        lambda x: (x == 'Yes').sum(), axis=1
    )
    
    # High value customer flag
    df['high_value'] = ((df['MonthlyCharges'] > df['MonthlyCharges'].quantile(0.75)) & 
                        (df['tenure'] > df['tenure'].median())).astype(int)
    
    logger.info("Created interaction features")
    return df


def encode_categorical_features(df: pd.DataFrame, target_col: str = 'Churn') -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Encode categorical features using one-hot encoding.
    
    Args:
        df: DataFrame with categorical features
        target_col: Name of target column
        
    Returns:
        Tuple of (encoded DataFrame, customer IDs)
    """
    logger.info("Encoding categorical features...")
    df_encoded = df.copy()
    
    # Separate target variable
    if target_col in df_encoded.columns:
        target = df_encoded[target_col]
        df_encoded = df_encoded.drop(columns=[target_col])
    else:
        target = None
    
    # Store customer IDs
    if 'customerID' in df_encoded.columns:
        customer_ids = df_encoded['customerID']
        df_encoded = df_encoded.drop(columns=['customerID'])
    else:
        customer_ids = None
    
    # Get categorical columns
    categorical_cols = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
    logger.info(f"Categorical columns to encode: {categorical_cols}")
    
    # One-hot encode categorical variables (drop_first to avoid multicollinearity)
    df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols, drop_first=True, dtype=int)
    
    # Add target back if it exists
    if target is not None:
        df_encoded[target_col] = target
    
    logger.info(f"Encoded features shape: {df_encoded.shape}")
    return df_encoded, customer_ids


def scale_features(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame,
                   scaler_type: str = 'standard', save_scaler: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Scale features using StandardScaler or RobustScaler.
    
    Args:
        X_train: Training features
        X_val: Validation features
        X_test: Test features
        scaler_type: Type of scaler ('standard' or 'robust')
        save_scaler: Whether to save the scaler
        
    Returns:
        Scaled X_train, X_val, X_test
    """
    logger.info(f"Scaling features using {scaler_type} scaler...")
    
    project_root = get_project_root()
    models_dir = project_root / "models"
    ensure_dir(models_dir)
    
    if scaler_type == 'robust':
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()
    
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        columns=X_val.columns,
        index=X_val.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    if save_scaler:
        scaler_path = models_dir / "scaler.pkl"
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler saved to {scaler_path}")
    
    return X_train_scaled, X_val_scaled, X_test_scaled


def prepare_data(df: pd.DataFrame, target_col: str = 'Churn',
                 test_size: float = 0.2, val_size: float = 0.1,
                 random_state: int = 42, scale_features_flag: bool = True,
                 scaler_type: str = 'standard') -> Tuple:
    """
    Complete advanced data preparation pipeline.
    
    Args:
        df: Input DataFrame
        target_col: Name of target column
        test_size: Proportion of test set
        val_size: Proportion of validation set (from training set)
        random_state: Random seed
        scale_features_flag: Whether to scale features
        scaler_type: Type of scaler to use
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names
    """
    logger.info("Starting advanced data preparation pipeline...")
    
    # Create advanced features
    df = create_tenure_bins(df)
    df = calculate_clv(df)
    df = create_interaction_features(df)
    
    # Encode features
    df_encoded, customer_ids = encode_categorical_features(df, target_col)
    
    # Separate features and target
    X = df_encoded.drop(columns=[target_col])
    y = df_encoded[target_col]
    
    # Encode target variable (Yes/No -> 1/0)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Save label encoder
    project_root = get_project_root()
    models_dir = project_root / "models"
    ensure_dir(models_dir)
    label_encoder_path = models_dir / "label_encoder.pkl"
    joblib.dump(label_encoder, label_encoder_path)
    logger.info(f"Label encoder saved to {label_encoder_path}")
    
    # Train-validation-test split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )
    
    val_size_adjusted = val_size / (1 - test_size)  # Adjust for already split data
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
    )
    
    logger.info(f"Train set shape: {X_train.shape}")
    logger.info(f"Validation set shape: {X_val.shape}")
    logger.info(f"Test set shape: {X_test.shape}")
    logger.info(f"Train churn rate: {y_train.mean():.2%}")
    logger.info(f"Val churn rate: {y_val.mean():.2%}")
    logger.info(f"Test churn rate: {y_test.mean():.2%}")
    
    # Scale features if requested
    if scale_features_flag:
        X_train, X_val, X_test = scale_features(X_train, X_val, X_test, scaler_type=scaler_type)
    
    feature_names = X_train.columns.tolist()
    logger.info(f"Total features: {len(feature_names)}")
    
    # Save processed data
    data_dir = project_root / "data"
    ensure_dir(data_dir)
    X_train.to_csv(data_dir / "X_train.csv", index=False)
    X_val.to_csv(data_dir / "X_val.csv", index=False)
    X_test.to_csv(data_dir / "X_test.csv", index=False)
    pd.Series(y_train).to_csv(data_dir / "y_train.csv", index=False)
    pd.Series(y_val).to_csv(data_dir / "y_val.csv", index=False)
    pd.Series(y_test).to_csv(data_dir / "y_test.csv", index=False)
    
    logger.info("Feature engineering completed successfully!")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names


def main():
    """Main function to run feature engineering pipeline."""
    # Load data
    df = load_cleaned_data()
    
    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = prepare_data(
        df, target_col='Churn', test_size=0.2, val_size=0.1,
        random_state=42, scale_features_flag=True, scaler_type='standard'
    )
    
    logger.info("Feature engineering completed successfully!")
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = main()
