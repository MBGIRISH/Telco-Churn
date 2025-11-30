"""
Advanced Data Cleaning Module
Handles missing values, duplicates, outliers, and data validation
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Optional
import sys

# Handle imports for both direct execution and module import
try:
    from src.utils import get_project_root, ensure_dir
except ImportError:
    # If running directly from src/, add parent to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.utils import get_project_root, ensure_dir

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data(file_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the dataset from CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame with loaded data
    """
    if file_path is None:
        project_root = get_project_root()
        file_path = project_root / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    
    logger.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    logger.info(f"Data loaded successfully. Shape: {df.shape}")
    return df


def detect_outliers_iqr(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Detect outliers using IQR method.
    
    Args:
        df: DataFrame
        columns: List of column names to check
        
    Returns:
        DataFrame with outlier flags
    """
    outlier_flags = pd.DataFrame(index=df.index)
    
    for col in columns:
        if col not in df.columns:
            continue
        
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_flags[f'{col}_outlier'] = (df[col] < lower_bound) | (df[col] > upper_bound)
    
    return outlier_flags


def clean_data(df: pd.DataFrame, remove_outliers: bool = False) -> Tuple[pd.DataFrame, dict]:
    """
    Clean the dataset comprehensively.
    
    Args:
        df: Raw DataFrame
        remove_outliers: Whether to remove outliers
        
    Returns:
        Tuple of (cleaned DataFrame, cleaning report)
    """
    logger.info("Starting advanced data cleaning process...")
    df_clean = df.copy()
    cleaning_report = {
        'initial_rows': len(df_clean),
        'initial_columns': len(df_clean.columns),
        'duplicates_removed': 0,
        'missing_values_handled': {},
        'outliers_removed': 0,
        'type_conversions': []
    }
    
    # Replace empty strings with NaN
    logger.info("Replacing empty strings with NaN...")
    df_clean.replace([" ", "", "nan", "None"], np.nan, inplace=True)
    
    # Check for missing values
    missing_values = df_clean.isnull().sum()
    if missing_values.sum() > 0:
        logger.info(f"Missing values found:\n{missing_values[missing_values > 0]}")
        cleaning_report['missing_values_handled'] = missing_values[missing_values > 0].to_dict()
    
    # Handle TotalCharges - convert to numeric and fill missing values
    if 'TotalCharges' in df_clean.columns:
        logger.info("Converting TotalCharges to numeric...")
        df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
        # Fill missing TotalCharges with 0 (for new customers with tenure=0)
        missing_count = df_clean['TotalCharges'].isnull().sum()
        df_clean['TotalCharges'].fillna(0, inplace=True)
        cleaning_report['missing_values_handled']['TotalCharges'] = missing_count
    
    # Drop rows with any remaining missing values
    initial_shape = df_clean.shape
    df_clean.dropna(inplace=True)
    dropped_rows = initial_shape[0] - df_clean.shape[0]
    logger.info(f"Dropped {dropped_rows} rows with missing values")
    
    # Remove duplicates
    initial_shape = df_clean.shape
    df_clean.drop_duplicates(inplace=True)
    duplicates_removed = initial_shape[0] - df_clean.shape[0]
    cleaning_report['duplicates_removed'] = duplicates_removed
    logger.info(f"Removed {duplicates_removed} duplicate rows")
    
    # Type conversions
    if 'SeniorCitizen' in df_clean.columns:
        df_clean['SeniorCitizen'] = df_clean['SeniorCitizen'].astype(int)
        cleaning_report['type_conversions'].append('SeniorCitizen -> int')
    
    if 'tenure' in df_clean.columns:
        df_clean['tenure'] = df_clean['tenure'].astype(int)
        cleaning_report['type_conversions'].append('tenure -> int')
    
    # Ensure MonthlyCharges and TotalCharges are float
    if 'MonthlyCharges' in df_clean.columns:
        df_clean['MonthlyCharges'] = df_clean['MonthlyCharges'].astype(float)
    if 'TotalCharges' in df_clean.columns:
        df_clean['TotalCharges'] = df_clean['TotalCharges'].astype(float)
    
    # Outlier detection and removal
    if remove_outliers:
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        if 'customerID' in numerical_cols:
            numerical_cols.remove('customerID')
        
        outlier_flags = detect_outliers_iqr(df_clean, numerical_cols)
        total_outliers = outlier_flags.any(axis=1).sum()
        
        if total_outliers > 0:
            logger.info(f"Detected {total_outliers} rows with outliers")
            # Remove rows with outliers in critical columns
            critical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
            critical_outlier_cols = [f'{col}_outlier' for col in critical_cols if f'{col}_outlier' in outlier_flags.columns]
            
            if critical_outlier_cols:
                rows_with_critical_outliers = outlier_flags[critical_outlier_cols].any(axis=1)
                df_clean = df_clean[~rows_with_critical_outliers]
                outliers_removed = rows_with_critical_outliers.sum()
                cleaning_report['outliers_removed'] = outliers_removed
                logger.info(f"Removed {outliers_removed} rows with outliers in critical columns")
    
    cleaning_report['final_rows'] = len(df_clean)
    cleaning_report['final_columns'] = len(df_clean.columns)
    
    logger.info(f"Data cleaning completed. Final shape: {df_clean.shape}")
    logger.info(f"Cleaning report: {cleaning_report}")
    
    return df_clean, cleaning_report


def main():
    """Main function to run data cleaning pipeline."""
    project_root = get_project_root()
    data_dir = project_root / "data"
    ensure_dir(data_dir)
    
    # Load and clean data
    df = load_data()
    df_clean, report = clean_data(df, remove_outliers=False)
    
    # Save cleaned data
    output_path = data_dir / "cleaned_data.csv"
    df_clean.to_csv(output_path, index=False)
    logger.info(f"Cleaned data saved to {output_path}")
    
    # Save cleaning report
    try:
        from src.utils import save_json
    except ImportError:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.utils import save_json
    
    report_path = project_root / "models" / "cleaning_report.json"
    ensure_dir(report_path.parent)
    save_json(report, report_path)
    
    return df_clean, report


if __name__ == "__main__":
    cleaned_df, cleaning_report = main()
