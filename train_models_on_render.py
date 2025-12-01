"""
Script to train models on Render if they don't exist
Run this in the build command or as a startup script
"""

import os
import sys
from pathlib import Path
import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_and_train_models():
    """Check if models exist, train if missing."""
    project_root = Path.cwd()
    models_dir = project_root / "models"
    
    # Check if any models exist
    model_files = list(models_dir.glob("*.pkl")) if models_dir.exists() else []
    
    if not model_files:
        logger.info("No models found. Training models...")
        logger.info("This may take several minutes...")
        
        try:
            # Run data cleaning
            logger.info("Step 1/4: Data cleaning...")
            subprocess.run([sys.executable, "src/data_cleaning.py"], check=True, cwd=project_root)
            
            # Run feature engineering
            logger.info("Step 2/4: Feature engineering...")
            subprocess.run([sys.executable, "src/feature_engineering.py"], check=True, cwd=project_root)
            
            # Run model training
            logger.info("Step 3/4: Training models...")
            subprocess.run([sys.executable, "src/model_training.py"], check=True, cwd=project_root)
            
            # Run PyTorch training (optional, can skip if takes too long)
            logger.info("Step 4/4: Training PyTorch model...")
            try:
                subprocess.run([sys.executable, "src/model_pytorch.py"], check=True, cwd=project_root, timeout=600)
            except subprocess.TimeoutExpired:
                logger.warning("PyTorch training timed out. Skipping...")
            except Exception as e:
                logger.warning(f"PyTorch training failed: {e}. Continuing...")
            
            logger.info("Model training completed!")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error training models: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return False
    else:
        logger.info(f"Models already exist ({len(model_files)} files). Skipping training.")
        return True

if __name__ == "__main__":
    check_and_train_models()

