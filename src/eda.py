"""
Exploratory Data Analysis (EDA) Module
Creates visualizations and insights using Matplotlib and Seaborn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def load_cleaned_data(file_path: str = None) -> pd.DataFrame:
    """Load cleaned data."""
    if file_path is None:
        project_root = Path(__file__).parent.parent
        file_path = project_root / "data" / "cleaned_data.csv"
    
    logger.info(f"Loading cleaned data from {file_path}")
    df = pd.read_csv(file_path)
    return df


def plot_churn_distribution(df: pd.DataFrame, save_path: str = None):
    """
    Plot churn distribution.
    
    Args:
        df: DataFrame with Churn column
        save_path: Path to save the figure
    """
    logger.info("Creating churn distribution plot...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Count plot
    sns.countplot(data=df, x="Churn", ax=axes[0], palette="Set2")
    axes[0].set_title("Churn Distribution (Count)", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Churn", fontsize=12)
    axes[0].set_ylabel("Count", fontsize=12)
    
    # Pie chart
    churn_counts = df['Churn'].value_counts()
    axes[1].pie(churn_counts.values, labels=churn_counts.index, autopct='%1.1f%%', 
                colors=['#66b3ff', '#ff9999'], startangle=90)
    axes[1].set_title("Churn Distribution (Percentage)", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
    plt.close()


def plot_correlation_matrix(df: pd.DataFrame, save_path: str = None):
    """
    Plot correlation matrix for numerical features.
    
    Args:
        df: DataFrame
        save_path: Path to save the figure
    """
    logger.info("Creating correlation matrix...")
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if 'customerID' in numerical_cols:
        numerical_cols.remove('customerID')
    
    corr_matrix = df[numerical_cols].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title("Correlation Matrix of Numerical Features", fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
    plt.close()


def plot_contract_vs_churn(df: pd.DataFrame, save_path: str = None):
    """
    Plot contract type vs churn.
    
    Args:
        df: DataFrame
        save_path: Path to save the figure
    """
    logger.info("Creating contract vs churn plot...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Count plot
    sns.countplot(data=df, x="Contract", hue="Churn", ax=axes[0], palette="Set2")
    axes[0].set_title("Contract Type vs Churn (Count)", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Contract Type", fontsize=12)
    axes[0].set_ylabel("Count", fontsize=12)
    axes[0].legend(title="Churn")
    axes[0].tick_params(axis='x', rotation=45)
    
    # Percentage stacked bar
    contract_churn = pd.crosstab(df['Contract'], df['Churn'], normalize='index') * 100
    contract_churn.plot(kind='bar', stacked=True, ax=axes[1], color=['#66b3ff', '#ff9999'])
    axes[1].set_title("Contract Type vs Churn (Percentage)", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Contract Type", fontsize=12)
    axes[1].set_ylabel("Percentage", fontsize=12)
    axes[1].legend(title="Churn")
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
    plt.close()


def plot_monthly_charges_vs_churn(df: pd.DataFrame, save_path: str = None):
    """
    Plot monthly charges vs churn.
    
    Args:
        df: DataFrame
        save_path: Path to save the figure
    """
    logger.info("Creating monthly charges vs churn plot...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Box plot
    sns.boxplot(data=df, x="Churn", y="MonthlyCharges", ax=axes[0], palette="Set2")
    axes[0].set_title("Monthly Charges Distribution by Churn", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Churn", fontsize=12)
    axes[0].set_ylabel("Monthly Charges", fontsize=12)
    
    # Violin plot
    sns.violinplot(data=df, x="Churn", y="MonthlyCharges", ax=axes[1], palette="Set2")
    axes[1].set_title("Monthly Charges Distribution by Churn (Violin)", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Churn", fontsize=12)
    axes[1].set_ylabel("Monthly Charges", fontsize=12)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
    plt.close()


def plot_tenure_vs_churn(df: pd.DataFrame, save_path: str = None):
    """
    Plot tenure vs churn.
    
    Args:
        df: DataFrame
        save_path: Path to save the figure
    """
    logger.info("Creating tenure vs churn plot...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Box plot
    sns.boxplot(data=df, x="Churn", y="tenure", ax=axes[0], palette="Set2")
    axes[0].set_title("Tenure Distribution by Churn", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Churn", fontsize=12)
    axes[0].set_ylabel("Tenure (months)", fontsize=12)
    
    # Histogram
    for churn_status in df['Churn'].unique():
        subset = df[df['Churn'] == churn_status]
        axes[1].hist(subset['tenure'], alpha=0.6, label=churn_status, bins=30)
    axes[1].set_title("Tenure Distribution by Churn", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Tenure (months)", fontsize=12)
    axes[1].set_ylabel("Frequency", fontsize=12)
    axes[1].legend(title="Churn")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
    plt.close()


def plot_payment_method_vs_churn(df: pd.DataFrame, save_path: str = None):
    """
    Plot payment method vs churn.
    
    Args:
        df: DataFrame
        save_path: Path to save the figure
    """
    logger.info("Creating payment method vs churn plot...")
    plt.figure(figsize=(12, 6))
    
    payment_churn = pd.crosstab(df['PaymentMethod'], df['Churn'], normalize='index') * 100
    payment_churn.plot(kind='bar', stacked=True, color=['#66b3ff', '#ff9999'])
    plt.title("Payment Method vs Churn", fontsize=14, fontweight='bold')
    plt.xlabel("Payment Method", fontsize=12)
    plt.ylabel("Percentage", fontsize=12)
    plt.legend(title="Churn")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
    plt.close()


def generate_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate summary statistics.
    
    Args:
        df: DataFrame
        
    Returns:
        Summary statistics DataFrame
    """
    logger.info("Generating summary statistics...")
    summary = df.describe(include='all')
    return summary


def main():
    """Main function to run EDA pipeline."""
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "notebooks" / "eda_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_cleaned_data()
    
    # Generate all plots
    plot_churn_distribution(df, save_path=str(output_dir / "churn_distribution.png"))
    plot_correlation_matrix(df, save_path=str(output_dir / "correlation_matrix.png"))
    plot_contract_vs_churn(df, save_path=str(output_dir / "contract_vs_churn.png"))
    plot_monthly_charges_vs_churn(df, save_path=str(output_dir / "monthly_charges_vs_churn.png"))
    plot_tenure_vs_churn(df, save_path=str(output_dir / "tenure_vs_churn.png"))
    plot_payment_method_vs_churn(df, save_path=str(output_dir / "payment_method_vs_churn.png"))
    
    # Generate summary statistics
    summary = generate_summary_statistics(df)
    summary_path = output_dir / "summary_statistics.csv"
    summary.to_csv(summary_path)
    logger.info(f"Summary statistics saved to {summary_path}")
    
    logger.info("EDA completed successfully!")
    return df


if __name__ == "__main__":
    df = main()



