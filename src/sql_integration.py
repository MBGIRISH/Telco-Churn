"""
SQL Integration Module
Creates SQL table and provides functions to fetch data from SQL and make predictions
"""

import pandas as pd
import numpy as np
import sqlite3
import logging
from pathlib import Path
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_sql_table(db_path: str = "telco_churn.db", table_name: str = "customers"):
    """
    Create SQL table for customer data.
    
    Args:
        db_path: Path to SQLite database file
        table_name: Name of the table
    """
    logger.info(f"Creating SQL table '{table_name}' in database '{db_path}'...")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create table
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        customerID TEXT PRIMARY KEY,
        gender TEXT,
        SeniorCitizen INTEGER,
        Partner TEXT,
        Dependents TEXT,
        tenure INTEGER,
        PhoneService TEXT,
        MultipleLines TEXT,
        InternetService TEXT,
        OnlineSecurity TEXT,
        OnlineBackup TEXT,
        DeviceProtection TEXT,
        TechSupport TEXT,
        StreamingTV TEXT,
        StreamingMovies TEXT,
        Contract TEXT,
        PaperlessBilling TEXT,
        PaymentMethod TEXT,
        MonthlyCharges REAL,
        TotalCharges REAL,
        Churn TEXT
    )
    """
    
    cursor.execute(create_table_query)
    conn.commit()
    logger.info(f"Table '{table_name}' created successfully")
    
    conn.close()


def insert_data_from_csv(csv_path: str, db_path: str = "telco_churn.db", 
                         table_name: str = "customers"):
    """
    Insert data from CSV file into SQL table.
    
    Args:
        csv_path: Path to CSV file
        db_path: Path to SQLite database file
        table_name: Name of the table
    """
    logger.info(f"Inserting data from {csv_path} into {table_name}...")
    
    df = pd.read_csv(csv_path)
    
    conn = sqlite3.connect(db_path)
    
    # Replace empty strings with None
    df = df.replace(' ', None)
    df = df.replace('', None)
    
    # Insert data
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    
    conn.close()
    logger.info(f"Data inserted successfully. Total rows: {len(df)}")


def fetch_customers_from_sql(db_path: str = "telco_churn.db", 
                             table_name: str = "customers",
                             query: str = None) -> pd.DataFrame:
    """
    Fetch customer data from SQL database.
    
    Args:
        db_path: Path to SQLite database file
        table_name: Name of the table
        query: Custom SQL query (optional)
        
    Returns:
        DataFrame with customer data
    """
    logger.info(f"Fetching data from {table_name}...")
    
    conn = sqlite3.connect(db_path)
    
    if query is None:
        query = f"SELECT * FROM {table_name}"
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    logger.info(f"Fetched {len(df)} rows from database")
    return df


def predict_churn_from_sql(db_path: str = "telco_churn.db",
                           table_name: str = "customers",
                           model_path: str = None,
                           scaler_path: str = None,
                           label_encoder_path: str = None,
                           query: str = None) -> pd.DataFrame:
    """
    Fetch data from SQL, preprocess, and make churn predictions.
    
    Args:
        db_path: Path to SQLite database file
        table_name: Name of the table
        model_path: Path to saved model
        scaler_path: Path to saved scaler
        label_encoder_path: Path to saved label encoder
        query: Custom SQL query (optional)
        
    Returns:
        DataFrame with predictions
    """
    logger.info("Predicting churn from SQL data...")
    
    # Load model and preprocessors
    project_root = Path(__file__).parent.parent
    
    if model_path is None:
        model_path = project_root / "models" / "xgboost.pkl"
    if scaler_path is None:
        scaler_path = project_root / "models" / "scaler.pkl"
    if label_encoder_path is None:
        label_encoder_path = project_root / "models" / "label_encoder.pkl"
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(label_encoder_path)
    
    # Fetch data from SQL
    df = fetch_customers_from_sql(db_path, table_name, query)
    
    if len(df) == 0:
        logger.warning("No data found in database")
        return pd.DataFrame()
    
    # Preprocess data (similar to feature_engineering.py)
    import sys
    from pathlib import Path as PathLib
    sys.path.append(str(PathLib(__file__).parent))
    from feature_engineering import encode_categorical_features
    
    df_encoded, customer_ids = encode_categorical_features(df, target_col='Churn')
    
    # Separate features
    if 'Churn' in df_encoded.columns:
        X = df_encoded.drop(columns=['Churn'])
    else:
        X = df_encoded
    
    # Get feature names from training data to ensure all columns are present
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    X_train_path = data_dir / "X_train.csv"
    if X_train_path.exists():
        X_train_ref = pd.read_csv(X_train_path)
        feature_names = X_train_ref.columns.tolist()
        
        # Ensure all features from training are present
        for col in feature_names:
            if col not in X.columns:
                X[col] = 0
        
        # Reorder columns to match training data
        X = X[feature_names]
    
    # Scale features
    X_scaled = pd.DataFrame(
        scaler.transform(X),
        columns=X.columns,
        index=X.index
    )
    
    # Make predictions
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]
    
    # Decode predictions
    predictions_decoded = label_encoder.inverse_transform(predictions)
    
    # Add predictions to dataframe
    result_df = df.copy()
    result_df['Predicted_Churn'] = predictions_decoded
    result_df['Churn_Probability'] = probabilities
    
    logger.info(f"Predictions completed for {len(result_df)} customers")
    logger.info(f"Predicted churn rate: {(predictions == 1).mean():.2%}")
    
    return result_df


def example_queries():
    """Example SQL queries for common use cases."""
    queries = {
        'high_risk_customers': """
            SELECT * FROM customers 
            WHERE tenure < 12 AND Contract = 'Month-to-month'
        """,
        'low_tenure_customers': """
            SELECT * FROM customers 
            WHERE tenure < 6
        """,
        'high_monthly_charges': """
            SELECT * FROM customers 
            WHERE MonthlyCharges > 80
        """,
        'month_to_month_contracts': """
            SELECT * FROM customers 
            WHERE Contract = 'Month-to-month'
        """
    }
    
    return queries


def main():
    """Main function to set up SQL integration."""
    project_root = Path(__file__).parent.parent
    
    # Create database and table
    db_path = project_root / "data" / "telco_churn.db"
    create_sql_table(str(db_path))
    
    # Insert data from cleaned CSV
    csv_path = project_root / "data" / "cleaned_data.csv"
    if csv_path.exists():
        insert_data_from_csv(str(csv_path), str(db_path))
    else:
        logger.warning(f"Cleaned data file not found at {csv_path}")
    
    # Example: Fetch and predict
    try:
        # Example query: customers with tenure < 12
        query = "SELECT * FROM customers WHERE tenure < 12 LIMIT 10"
        predictions_df = predict_churn_from_sql(
            db_path=str(db_path),
            query=query
        )
        
        if len(predictions_df) > 0:
            logger.info("\nSample predictions:")
            logger.info(predictions_df[['customerID', 'tenure', 'MonthlyCharges', 
                                       'Predicted_Churn', 'Churn_Probability']].head())
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        logger.info("Make sure models are trained first by running model_training.py")
    
    logger.info("\nSQL integration setup completed!")
    return db_path


if __name__ == "__main__":
    db_path = main()



