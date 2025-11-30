"""
FastAPI REST API for Customer Churn Prediction
Provides endpoints for predictions and customer data
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import pandas as pd
import numpy as np
import joblib
import torch
import logging
from pathlib import Path
from datetime import datetime
import sys
from pathlib import Path

# Handle imports
try:
    from src.utils import get_project_root
    from src.database import fetch_customer_data, insert_prediction, get_customer_predictions
    from src.deployment_streamlit import preprocess_input, predict_churn
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.utils import get_project_root
    from src.database import fetch_customer_data, insert_prediction, get_customer_predictions
    from src.deployment_streamlit import preprocess_input, predict_churn

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="REST API for predicting customer churn",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models at startup
project_root = get_project_root()
models_dir = project_root / "models"

ml_model = None
pytorch_model = None
scaler = None
label_encoder = None
feature_names = None


@app.on_event("startup")
async def load_models():
    """Load models at application startup."""
    global ml_model, pytorch_model, scaler, label_encoder, feature_names
    
    try:
        # Load ML model
        ml_model_path = models_dir / "best_model.pkl"
        if not ml_model_path.exists():
            ml_model_path = models_dir / "random_forest.pkl"
        
        if ml_model_path.exists():
            ml_model = joblib.load(ml_model_path)
            logger.info(f"Loaded ML model: {ml_model_path.name}")
        
        # Load PyTorch model
        pytorch_model_path = models_dir / "best_pytorch_model.pth"
        if pytorch_model_path.exists():
            checkpoint = torch.load(pytorch_model_path, map_location='cpu')
            from src.model_pytorch import AdvancedANNModel
            pytorch_model = AdvancedANNModel(
                input_dim=checkpoint['input_dim'],
                hidden_layers=checkpoint['hidden_layers'],
                dropout_rate=checkpoint['dropout_rate'],
                use_batch_norm=checkpoint['use_batch_norm']
            )
            pytorch_model.load_state_dict(checkpoint['model_state_dict'])
            pytorch_model.eval()
            logger.info(f"Loaded PyTorch model: {pytorch_model_path.name}")
        
        # Load preprocessors
        scaler_path = models_dir / "scaler.pkl"
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
        
        label_encoder_path = models_dir / "label_encoder.pkl"
        if label_encoder_path.exists():
            label_encoder = joblib.load(label_encoder_path)
        
        # Load feature names
        X_train_path = project_root / "data" / "X_train.csv"
        if X_train_path.exists():
            X_train = pd.read_csv(X_train_path)
            feature_names = X_train.columns.tolist()
        
        logger.info("All models and preprocessors loaded successfully")
    except Exception as e:
        logger.error(f"Error loading models: {e}")


# Pydantic models for request/response
class CustomerInput(BaseModel):
    """Customer input for prediction."""
    gender: str = Field(..., description="Gender: Male or Female")
    SeniorCitizen: int = Field(..., ge=0, le=1, description="Senior Citizen: 0 or 1")
    Partner: str = Field(..., description="Partner: Yes or No")
    Dependents: str = Field(..., description="Dependents: Yes or No")
    tenure: int = Field(..., ge=0, le=72, description="Tenure in months")
    PhoneService: str = Field(..., description="Phone Service: Yes or No")
    MultipleLines: str = Field(..., description="Multiple Lines: Yes, No, or No phone service")
    InternetService: str = Field(..., description="Internet Service: DSL, Fiber optic, or No")
    OnlineSecurity: str = Field(..., description="Online Security: Yes, No, or No internet service")
    OnlineBackup: str = Field(..., description="Online Backup: Yes, No, or No internet service")
    DeviceProtection: str = Field(..., description="Device Protection: Yes, No, or No internet service")
    TechSupport: str = Field(..., description="Tech Support: Yes, No, or No internet service")
    StreamingTV: str = Field(..., description="Streaming TV: Yes, No, or No internet service")
    StreamingMovies: str = Field(..., description="Streaming Movies: Yes, No, or No internet service")
    Contract: str = Field(..., description="Contract: Month-to-month, One year, or Two year")
    PaperlessBilling: str = Field(..., description="Paperless Billing: Yes or No")
    PaymentMethod: str = Field(..., description="Payment Method")
    MonthlyCharges: float = Field(..., ge=0, description="Monthly Charges")
    TotalCharges: float = Field(..., ge=0, description="Total Charges")


class PredictionResponse(BaseModel):
    """Prediction response."""
    predicted_churn: str = Field(..., description="Predicted churn: Yes or No")
    churn_probability: float = Field(..., ge=0, le=1, description="Churn probability")
    model_used: str = Field(..., description="Model used for prediction")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Customer Churn Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Predict churn using ML model",
            "/predict-pytorch": "POST - Predict churn using PyTorch model",
            "/customer/{customer_id}": "GET - Get customer data",
            "/health": "GET - Health check"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "ml_model_loaded": ml_model is not None,
        "pytorch_model_loaded": pytorch_model is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_churn_ml(customer: CustomerInput):
    """
    Predict churn using ML model.
    
    Args:
        customer: Customer input data
        
    Returns:
        Prediction result
    """
    if ml_model is None or scaler is None or label_encoder is None:
        raise HTTPException(status_code=500, detail="ML model not loaded")
    
    try:
        # Preprocess input
        X_scaled = preprocess_input(
            customer.gender, customer.SeniorCitizen, customer.Partner, customer.Dependents,
            customer.tenure, customer.PhoneService, customer.MultipleLines,
            customer.InternetService, customer.OnlineSecurity, customer.OnlineBackup,
            customer.DeviceProtection, customer.TechSupport, customer.StreamingTV,
            customer.StreamingMovies, customer.Contract, customer.PaperlessBilling,
            customer.PaymentMethod, customer.MonthlyCharges, customer.TotalCharges,
            scaler, feature_names
        )
        
        # Make prediction
        prediction, probability = predict_churn(ml_model, X_scaled)
        predicted_churn = label_encoder.inverse_transform([prediction])[0]
        
        return PredictionResponse(
            predicted_churn=predicted_churn,
            churn_probability=float(probability),
            model_used="ML Model"
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict-pytorch", response_model=PredictionResponse)
async def predict_churn_pytorch(customer: CustomerInput):
    """
    Predict churn using PyTorch model.
    
    Args:
        customer: Customer input data
        
    Returns:
        Prediction result
    """
    if pytorch_model is None or scaler is None or label_encoder is None:
        raise HTTPException(status_code=500, detail="PyTorch model not loaded")
    
    try:
        # Preprocess input
        X_scaled = preprocess_input(
            customer.gender, customer.SeniorCitizen, customer.Partner, customer.Dependents,
            customer.tenure, customer.PhoneService, customer.MultipleLines,
            customer.InternetService, customer.OnlineSecurity, customer.OnlineBackup,
            customer.DeviceProtection, customer.TechSupport, customer.StreamingTV,
            customer.StreamingMovies, customer.Contract, customer.PaperlessBilling,
            customer.PaymentMethod, customer.MonthlyCharges, customer.TotalCharges,
            scaler, feature_names
        )
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_scaled.values)
        
        # Make prediction
        pytorch_model.eval()
        with torch.no_grad():
            output = pytorch_model(X_tensor)
            probability = output.item()
            prediction = 1 if probability > 0.5 else 0
        
        predicted_churn = label_encoder.inverse_transform([prediction])[0]
        
        return PredictionResponse(
            predicted_churn=predicted_churn,
            churn_probability=float(probability),
            model_used="PyTorch Model"
        )
    except Exception as e:
        logger.error(f"PyTorch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/customer/{customer_id}")
async def get_customer(customer_id: str):
    """
    Get customer data by ID.
    
    Args:
        customer_id: Customer ID
        
    Returns:
        Customer data
    """
    try:
        df = fetch_customer_data(customer_id=customer_id)
        if len(df) == 0:
            raise HTTPException(status_code=404, detail="Customer not found")
        
        customer_data = df.iloc[0].to_dict()
        return customer_data
    except Exception as e:
        logger.error(f"Error fetching customer: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/customer/{customer_id}/predictions")
async def get_customer_predictions_history(customer_id: str):
    """
    Get prediction history for a customer.
    
    Args:
        customer_id: Customer ID
        
    Returns:
        Prediction history
    """
    try:
        df = get_customer_predictions(customer_id)
        return df.to_dict(orient='records')
    except Exception as e:
        logger.error(f"Error fetching predictions: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

