"""
Render Deployment Setup Script
Runs on first deployment to initialize the environment
"""

import os
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_render_environment():
    """Setup environment for Render deployment."""
    logger.info("Setting up Render environment...")
    
    # Get project root
    project_root = Path(__file__).parent.parent
    
    # Create necessary directories
    directories = [
        project_root / "data",
        project_root / "models",
        project_root / "notebooks" / "evaluation_plots",
        project_root / "notebooks" / "shap_plots"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created/verified directory: {directory}")
    
    # Check if we need to run the pipeline
    models_dir = project_root / "models"
    data_dir = project_root / "data"
    
    # Check if models exist
    model_files = list(models_dir.glob("*.pkl")) + list(models_dir.glob("*.pth"))
    
    if not model_files:
        logger.warning("No models found. You may need to run the training pipeline.")
        logger.info("To train models, run: python src/model_training.py")
    else:
        logger.info(f"Found {len(model_files)} model files")
    
    # Check if database exists
    db_path = data_dir / "telco_churn.db"
    if not db_path.exists():
        logger.info("Database not found. Will be created on first use.")
    else:
        logger.info("Database found.")
    
    logger.info("Render environment setup complete!")
    return True

if __name__ == "__main__":
    setup_render_environment()

