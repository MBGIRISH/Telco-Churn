"""
Advanced SQL Database Integration Module
Handles database operations for customer data and predictions
"""

import pandas as pd
import numpy as np
import sqlite3
import logging
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime
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


def create_database(db_path: Optional[Path] = None, table_name: str = "customers"):
    """
    Create SQLite database and customers table.
    
    Args:
        db_path: Path to database file
        table_name: Name of the table
    """
    if db_path is None:
        project_root = get_project_root()
        db_path = project_root / "data" / "telco_churn.db"
    
    ensure_dir(db_path.parent)
    
    logger.info(f"Creating database at {db_path}...")
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Create customers table
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
        Churn TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """
    
    cursor.execute(create_table_query)
    
    # Create predictions table
    create_predictions_table = """
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        customerID TEXT,
        model_name TEXT,
        predicted_churn TEXT,
        churn_probability REAL,
        prediction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (customerID) REFERENCES customers(customerID)
    )
    """
    
    cursor.execute(create_predictions_table)
    
    # Create indexes
    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_tenure ON {table_name}(tenure)")
    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_contract ON {table_name}(Contract)")
    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_churn ON {table_name}(Churn)")
    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_monthly_charges ON {table_name}(MonthlyCharges)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_customer ON predictions(customerID)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_timestamp ON predictions(prediction_timestamp)")
    
    conn.commit()
    conn.close()
    
    logger.info(f"Database and tables created successfully")


def insert_customer_data(df: pd.DataFrame, db_path: Optional[Path] = None, 
                        table_name: str = "customers"):
    """
    Insert customer data into database.
    
    Args:
        df: DataFrame with customer data
        db_path: Path to database file
        table_name: Name of the table
    """
    if db_path is None:
        project_root = get_project_root()
        db_path = project_root / "data" / "telco_churn.db"
    
    logger.info(f"Inserting {len(df)} customers into database...")
    
    conn = sqlite3.connect(str(db_path))
    
    # Replace empty strings with None
    df = df.replace(' ', None)
    df = df.replace('', None)
    
    # Insert data (replace if exists)
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    
    conn.close()
    logger.info(f"Data inserted successfully. Total rows: {len(df)}")


def fetch_customer_data(db_path: Optional[Path] = None, table_name: str = "customers",
                       query: Optional[str] = None, customer_id: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch customer data from database.
    
    Args:
        db_path: Path to database file
        table_name: Name of the table
        query: Custom SQL query
        customer_id: Specific customer ID to fetch
        
    Returns:
        DataFrame with customer data
    """
    if db_path is None:
        project_root = get_project_root()
        db_path = project_root / "data" / "telco_churn.db"
    
    conn = sqlite3.connect(str(db_path))
    
    if query:
        df = pd.read_sql_query(query, conn)
    elif customer_id:
        df = pd.read_sql_query(
            f"SELECT * FROM {table_name} WHERE customerID = ?", 
            conn, params=(customer_id,)
        )
    else:
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    
    conn.close()
    logger.info(f"Fetched {len(df)} rows from database")
    return df


def insert_prediction(customer_id: str, model_name: str, predicted_churn: str,
                     churn_probability: float, db_path: Optional[Path] = None):
    """
    Insert prediction into predictions table.
    
    Args:
        customer_id: Customer ID
        model_name: Name of the model used
        predicted_churn: Prediction (Yes/No)
        churn_probability: Churn probability
        db_path: Path to database file
    """
    if db_path is None:
        project_root = get_project_root()
        db_path = project_root / "data" / "telco_churn.db"
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    insert_query = """
    INSERT INTO predictions (customerID, model_name, predicted_churn, churn_probability)
    VALUES (?, ?, ?, ?)
    """
    
    cursor.execute(insert_query, (customer_id, model_name, predicted_churn, churn_probability))
    conn.commit()
    conn.close()
    
    logger.info(f"Prediction inserted for customer {customer_id}")


def get_customer_predictions(customer_id: str, db_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Get prediction history for a customer.
    
    Args:
        customer_id: Customer ID
        db_path: Path to database file
        
    Returns:
        DataFrame with prediction history
    """
    if db_path is None:
        project_root = get_project_root()
        db_path = project_root / "data" / "telco_churn.db"
    
    conn = sqlite3.connect(str(db_path))
    
    query = """
    SELECT * FROM predictions 
    WHERE customerID = ? 
    ORDER BY prediction_timestamp DESC
    """
    
    df = pd.read_sql_query(query, conn, params=(customer_id,))
    conn.close()
    
    return df


def get_high_risk_customers(db_path: Optional[Path] = None, 
                           min_probability: float = 0.7) -> pd.DataFrame:
    """
    Get high-risk customers from predictions.
    
    Args:
        db_path: Path to database file
        min_probability: Minimum churn probability
        
    Returns:
        DataFrame with high-risk customers
    """
    if db_path is None:
        project_root = get_project_root()
        db_path = project_root / "data" / "telco_churn.db"
    
    conn = sqlite3.connect(str(db_path))
    
    query = """
    SELECT p.*, c.tenure, c.Contract, c.MonthlyCharges, c.Churn
    FROM predictions p
    JOIN customers c ON p.customerID = c.customerID
    WHERE p.churn_probability >= ?
    ORDER BY p.churn_probability DESC
    """
    
    df = pd.read_sql_query(query, conn, params=(min_probability,))
    conn.close()
    
    logger.info(f"Found {len(df)} high-risk customers (probability >= {min_probability})")
    return df


def get_statistics(db_path: Optional[Path] = None) -> Dict:
    """
    Get database statistics.
    
    Args:
        db_path: Path to database file
        
    Returns:
        Dictionary with statistics
    """
    if db_path is None:
        project_root = get_project_root()
        db_path = project_root / "data" / "telco_churn.db"
    
    conn = sqlite3.connect(str(db_path))
    
    stats = {}
    
    # Customer statistics
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM customers")
    stats['total_customers'] = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM customers WHERE Churn = 'Yes'")
    stats['churned_customers'] = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM predictions")
    stats['total_predictions'] = cursor.fetchone()[0]
    
    # Churn rate
    if stats['total_customers'] > 0:
        stats['churn_rate'] = stats['churned_customers'] / stats['total_customers']
    
    conn.close()
    
    return stats


def main():
    """Main function to set up database."""
    project_root = get_project_root()
    db_path = project_root / "data" / "telco_churn.db"
    
    # Create database
    create_database(db_path)
    
    # Insert data from cleaned CSV if available
    csv_path = project_root / "data" / "cleaned_data.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        insert_customer_data(df, db_path)
    else:
        logger.warning(f"Cleaned data file not found at {csv_path}")
    
    # Get statistics
    stats = get_statistics(db_path)
    logger.info(f"Database statistics: {stats}")
    
    logger.info("\nDatabase setup completed!")
    return db_path


if __name__ == "__main__":
    db_path = main()

