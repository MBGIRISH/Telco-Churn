"""
Advanced Streamlit App for Customer Churn Prediction
Includes SHAP explainability, model comparison, and PyTorch predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import logging
import os

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Render deployment setup (runs once)
if os.getenv("RENDER") or os.getenv("RENDER_EXTERNAL_URL"):
    try:
        from src.render_setup import setup_render_environment
        setup_render_environment()
    except Exception as e:
        logging.warning(f"Render setup skipped: {e}")

from deployment_streamlit import (
    load_model_and_preprocessors,
    preprocess_input,
    predict_churn,
    get_feature_importance
)
from src.model_pytorch import ChurnDataset, AdvancedANNModel
import torch
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction System - Advanced",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with dark theme support
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 2rem 0;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin: 2rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    
    /* Dark theme improvements */
    [data-testid="stAppViewContainer"] {
        background-color: #0e1117;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
            padding: 1rem 0;
        }
    }
    
    /* Card styling */
    .stMetric {
        background-color: rgba(28, 131, 225, 0.1);
        border: 1px solid rgba(28, 131, 225, 0.2);
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Better spacing */
    .element-container {
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">📊 Customer Churn Prediction System - Advanced</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
st.sidebar.title("🔧 Configuration")
model_option = st.sidebar.selectbox(
    "Select ML Model",
    ["random_forest", "xgboost", "lightgbm", "logistic_regression"],
    index=0
)

use_pytorch = st.sidebar.checkbox("Use PyTorch Model", value=False)

# Load models
@st.cache_resource
def load_ml_model_cached(model_name):
    try:
        return load_model_and_preprocessors(model_name)
    except FileNotFoundError as e:
        error_msg = str(e)
        st.error(f"❌ Model not found: {error_msg}")
        st.warning("""
        **Models are missing!** 
        
        To fix this on Render:
        
        1. **Option A - Train on Render** (Add to build command):
           ```bash
           pip install -r requirements.txt && python train_models_on_render.py
           ```
        
        2. **Option B - Upload models to GitHub**:
           - Train models locally: `python src/model_training.py`
           - Commit models to GitHub
           - Redeploy on Render
        
        3. **Option C - Use Render Shell**:
           - Go to Render dashboard → Your service → Shell
           - Run: `python src/model_training.py`
        """)
        return None, None, None, None
    except Exception as e:
        st.error(f"Error loading ML model: {e}")
        return None, None, None, None

@st.cache_resource
def load_pytorch_model():
    try:
        project_root = Path(__file__).parent
        models_dir = project_root / "models"
        model_path = models_dir / "best_pytorch_model.pth"
        
        if not model_path.exists():
            return None, None, None
        
        checkpoint = torch.load(model_path, map_location='cpu')
        model = AdvancedANNModel(
            input_dim=checkpoint['input_dim'],
            hidden_layers=checkpoint['hidden_layers'],
            dropout_rate=checkpoint['dropout_rate'],
            use_batch_norm=checkpoint['use_batch_norm']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Load scaler and label encoder
        scaler = joblib.load(models_dir / "scaler.pkl")
        label_encoder = joblib.load(models_dir / "label_encoder.pkl")
        
        return model, scaler, label_encoder
    except Exception as e:
        st.warning(f"PyTorch model not available: {e}")
        return None, None, None

ml_model, scaler, label_encoder, feature_names = load_ml_model_cached(model_option)
pytorch_model, pytorch_scaler, pytorch_label_encoder = load_pytorch_model()

if ml_model is None and not use_pytorch:
    st.error("Please train models first by running: python src/model_training.py")
    st.stop()

# Main content
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Prediction", "📈 Feature Importance", "🧠 SHAP Explainability", 
    "📊 Model Comparison"
])

with tab1:
    st.header("Customer Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
        Partner = st.selectbox("Partner", ["Yes", "No"])
        Dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    
    with col2:
        MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    
    with col3:
        StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
        PaymentMethod = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        MonthlyCharges = st.slider("Monthly Charges ($)", 18.0, 120.0, 50.0)
        TotalCharges = st.number_input("Total Charges ($)", min_value=0.0, value=500.0)
    
    # Predict button
    if st.button("🔮 Predict Churn", type="primary", use_container_width=True):
        try:
            if use_pytorch and pytorch_model is not None:
                # Use PyTorch model
                X_scaled = preprocess_input(
                    gender, SeniorCitizen, Partner, Dependents, tenure,
                    PhoneService, MultipleLines, InternetService, OnlineSecurity,
                    OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
                    StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
                    MonthlyCharges, TotalCharges, pytorch_scaler, feature_names
                )
                
                X_tensor = torch.FloatTensor(X_scaled.values)
                pytorch_model.eval()
                with torch.no_grad():
                    output = pytorch_model(X_tensor)
                    probability = output.item()
                    prediction = 1 if probability > 0.5 else 0
                
                churn_status = pytorch_label_encoder.inverse_transform([prediction])[0]
                model_used = "PyTorch ANN"
            else:
                # Use ML model
                X_scaled = preprocess_input(
                    gender, SeniorCitizen, Partner, Dependents, tenure,
                    PhoneService, MultipleLines, InternetService, OnlineSecurity,
                    OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
                    StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
                    MonthlyCharges, TotalCharges, scaler, feature_names
                )
                
                prediction, probability = predict_churn(ml_model, X_scaled)
                churn_status = label_encoder.inverse_transform([prediction])[0]
                model_used = model_option.upper()
            
            # Display results
            st.markdown("---")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                if churn_status == "Yes":
                    st.markdown(f"""
                        <div class="prediction-box">
                            <h2>⚠️ HIGH RISK - Customer Will Churn</h2>
                            <h1 style="font-size: 4rem; margin: 1rem 0;">{probability:.1%}</h1>
                            <p style="font-size: 1.2rem;">Churn Probability</p>
                            <p style="font-size: 1rem; margin-top: 1rem;">Model: {model_used}</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="prediction-box" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);">
                            <h2>✅ LOW RISK - Customer Will Stay</h2>
                            <h1 style="font-size: 4rem; margin: 1rem 0;">{1-probability:.1%}</h1>
                            <p style="font-size: 1.2rem;">Retention Probability</p>
                            <p style="font-size: 1rem; margin-top: 1rem;">Model: {model_used}</p>
                        </div>
                    """, unsafe_allow_html=True)
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Churn Probability", f"{probability:.2%}")
            with col2:
                st.metric("Retention Probability", f"{1-probability:.2%}")
            with col3:
                st.metric("Prediction", churn_status)
            with col4:
                st.metric("Model Used", model_used)
            
            # Probability gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = probability * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Churn Risk Score"},
                delta = {'reference': 50},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            st.exception(e)

with tab2:
    st.header("Feature Importance Analysis")
    
    try:
        if ml_model is None:
            st.warning("Please load a model first")
        else:
            importance_df = get_feature_importance(ml_model, feature_names, top_n=15)
            
            if importance_df is not None and len(importance_df) > 0:
                # Bar chart
                fig = px.bar(
                    importance_df,
                    x='importance',
                    y='feature',
                    orientation='h',
                    title="Top 15 Most Important Features",
                    labels={'importance': 'Importance Score', 'feature': 'Feature'},
                    color='importance',
                    color_continuous_scale='Blues'
                )
                fig.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Table
                st.subheader("Feature Importance Table")
                st.dataframe(importance_df, use_container_width=True)
            else:
                st.info("Feature importance not available for this model type.")
    except Exception as e:
        st.error(f"Error loading feature importance: {e}")

with tab3:
    st.header("🧠 SHAP Explainability")
    
    # Add explanation about SHAP
    with st.expander("ℹ️ What is SHAP?", expanded=False):
        st.markdown("""
        **SHAP (SHapley Additive exPlanations)** is a unified approach to explain the output of any machine learning model.
        
        - **Feature Importance**: Shows which features most influence predictions
        - **Feature Impact**: Positive/negative contributions to each prediction
        - **Model Interpretability**: Helps understand why a model makes specific predictions
        
        Higher SHAP values indicate features that push the prediction toward churn, while negative values push toward retention.
        """)
    
    try:
        from src.shap_utils import compute_shap_analysis, SHAP_AVAILABLE, export_shap_to_csv
        import shap
        
        if not SHAP_AVAILABLE:
            st.warning("⚠️ SHAP is not installed. Install with: pip install shap")
            st.code("pip install shap")
        else:
            # Configuration section
            col1, col2 = st.columns([1, 1])
            with col1:
                max_samples = st.slider("Sample Size", 20, 200, 100, help="More samples = more accurate but slower")
            with col2:
                selected_model = st.selectbox("Select Model", 
                                            ["best_model", "xgboost", "random_forest", "lightgbm", "logistic_regression"],
                                            index=0)
            
            # Compute SHAP button
            if st.button("🔍 Generate SHAP Explanations", type="primary", use_container_width=True):
                with st.spinner("🔄 Calculating SHAP values (this may take a minute)..."):
                    results = compute_shap_analysis(model_name=selected_model, max_samples=max_samples)
                    
                    if results:
                        st.session_state['shap_results'] = results
                        st.success("✅ SHAP analysis completed!")
            
            # Display results if available
            if 'shap_results' in st.session_state:
                results = st.session_state['shap_results']
                
                # Summary metrics cards
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Features Analyzed", len(results['feature_names']))
                with col2:
                    st.metric("Samples Used", len(results['X_sample']))
                with col3:
                    top_feat = results['summary_df'].iloc[0]['feature']
                    st.metric("Top Feature", top_feat[:25] + "..." if len(top_feat) > 25 else top_feat)
                with col4:
                    st.metric("Top Importance", f"{results['summary_df'].iloc[0]['shap_importance']:.4f}")
                
                st.markdown("---")
                
                # Tabs for different visualizations
                tab_summary, tab_single, tab_table = st.tabs(["📊 Summary", "🎯 Single Prediction", "📋 SHAP Table"])
                
                with tab_summary:
                    st.subheader("Global Feature Importance")
                    
                    # Beeswarm plot
                    st.markdown("#### Beeswarm Plot")
                    st.caption("Shows the distribution of SHAP values for each feature across all samples")
                    
                    try:
                        explanation = shap.Explanation(
                            values=results['shap_values'],
                            data=results['X_sample'].values if isinstance(results['X_sample'], pd.DataFrame) else results['X_sample'],
                            feature_names=results['feature_names']
                        )
                        st_shap = shap.plots.beeswarm(explanation, show=False)
                        # Note: SHAP plots need to be displayed using st.pyplot or saved as images
                        # For now, we'll use the bar chart instead
                    except Exception as e:
                        st.info("Beeswarm plot requires matplotlib. Showing bar chart instead.")
                    
                    # Mean SHAP importance bar chart
                    st.markdown("#### Mean SHAP Importance")
                    top_n = st.slider("Show Top N Features", 5, 30, 15, key="shap_top_n")
                    top_features = results['summary_df'].head(top_n)
                    
                    fig_importance = go.Figure()
                    fig_importance.add_trace(go.Bar(
                        x=top_features['shap_importance'],
                        y=top_features['feature'],
                        orientation='h',
                        marker=dict(
                            color=top_features['shap_importance'],
                            colorscale='RdYlGn_r',
                            showscale=True,
                            colorbar=dict(title="SHAP Importance")
                        ),
                        text=[f"{val:.4f}" for val in top_features['shap_importance']],
                        textposition='outside',
                        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>"
                    ))
                    
                    fig_importance.update_layout(
                        title=f"Top {top_n} Most Important Features (Mean |SHAP Value|)",
                        xaxis_title="Mean |SHAP Value|",
                        yaxis_title="Feature",
                        height=max(500, top_n * 35),
                        yaxis=dict(autorange="reversed"),
                        margin=dict(l=200, r=50),
                        template='plotly_dark' if st.get_option('theme.base') == 'dark' else 'plotly_white'
                    )
                    
                    st.plotly_chart(fig_importance, use_container_width=True)
                
                with tab_single:
                    st.subheader("Single Prediction Explanation")
                    
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        sample_idx = st.number_input("Sample Index", 0, len(results['X_sample']) - 1, 0,
                                                     help="Select which sample to explain")
                    
                    # Waterfall plot data
                    sample_shap = results['shap_values'][sample_idx]
                    sample_data = results['X_sample'].iloc[sample_idx] if isinstance(results['X_sample'], pd.DataFrame) else results['X_sample'][sample_idx]
                    
                    # Create waterfall-style visualization
                    st.markdown("#### Waterfall Plot - Feature Contributions")
                    
                    # Get top contributing features
                    contrib_df = pd.DataFrame({
                        'feature': results['feature_names'],
                        'shap_value': sample_shap,
                        'feature_value': sample_data.values if hasattr(sample_data, 'values') else sample_data
                    }).sort_values('shap_value', key=abs, ascending=False)
                    
                    top_contrib = contrib_df.head(15)
                    
                    # Waterfall chart
                    fig_waterfall = go.Figure()
                    
                    # Positive contributions
                    pos_contrib = top_contrib[top_contrib['shap_value'] > 0]
                    if len(pos_contrib) > 0:
                        fig_waterfall.add_trace(go.Bar(
                            x=pos_contrib['shap_value'],
                            y=pos_contrib['feature'],
                            orientation='h',
                            name='Positive Impact',
                            marker_color='#ff4444',
                            text=[f"+{val:.4f}" for val in pos_contrib['shap_value']],
                            textposition='outside'
                        ))
                    
                    # Negative contributions
                    neg_contrib = top_contrib[top_contrib['shap_value'] < 0]
                    if len(neg_contrib) > 0:
                        fig_waterfall.add_trace(go.Bar(
                            x=neg_contrib['shap_value'],
                            y=neg_contrib['feature'],
                            orientation='h',
                            name='Negative Impact',
                            marker_color='#44ff44',
                            text=[f"{val:.4f}" for val in neg_contrib['shap_value']],
                            textposition='outside'
                        ))
                    
                    fig_waterfall.update_layout(
                        title=f"Feature Contributions for Sample {sample_idx}",
                        xaxis_title="SHAP Value (Impact on Prediction)",
                        yaxis_title="Feature",
                        height=600,
                        yaxis=dict(autorange="reversed"),
                        margin=dict(l=200, r=50),
                        barmode='overlay',
                        template='plotly_dark' if st.get_option('theme.base') == 'dark' else 'plotly_white'
                    )
                    
                    st.plotly_chart(fig_waterfall, use_container_width=True)
                    
                    # Prediction summary
                    base_val = results['base_value']
                    prediction = float(base_val + sample_shap.sum())
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Base Value", f"{base_val:.4f}")
                    with col2:
                        st.metric("Total Contribution", f"{sample_shap.sum():.4f}")
                    with col3:
                        st.metric("Final Prediction", f"{prediction:.4f}", 
                                 f"{prediction*100:.1f}% churn probability")
                    
                    # Force plot data (for display)
                    st.markdown("#### Feature Values for This Sample")
                    st.dataframe(
                        contrib_df[['feature', 'feature_value', 'shap_value']].head(20),
                        use_container_width=True,
                        height=400
                    )
                
                with tab_table:
                    st.subheader("Complete SHAP Values Table")
                    
                    # Search box
                    search_term = st.text_input("🔍 Search Features", "", key="shap_search")
                    
                    # Create full SHAP table
                    shap_table = pd.DataFrame(
                        results['shap_values'],
                        columns=results['feature_names']
                    )
                    
                    # Add feature values
                    X_array = results['X_sample'].values if isinstance(results['X_sample'], pd.DataFrame) else results['X_sample']
                    for i, feat in enumerate(results['feature_names']):
                        shap_table[f'{feat}_value'] = X_array[:, i]
                    
                    shap_table.insert(0, 'sample_index', range(len(shap_table)))
                    
                    # Filter by search
                    if search_term:
                        mask = shap_table['sample_index'].astype(str).str.contains(search_term, case=False, na=False)
                        for col in results['feature_names']:
                            mask |= shap_table[col].astype(str).str.contains(search_term, case=False, na=False)
                        shap_table = shap_table[mask]
                    
                    st.dataframe(shap_table, use_container_width=True, height=500)
                    
                    # Download button
                    csv_data = export_shap_to_csv(
                        results['shap_values'],
                        results['X_sample'],
                        results['feature_names']
                    )
                    st.download_button(
                        label="📥 Download SHAP Values (CSV)",
                        data=csv_data,
                        file_name=f"shap_values_{selected_model}.csv",
                        mime="text/csv"
                    )
    except Exception as e:
        st.error(f"❌ Error in SHAP analysis: {e}")
        st.exception(e)

with tab4:
    st.header("📊 Model Comparison")
    
    # Add explanation
    with st.expander("ℹ️ Understanding Model Metrics", expanded=False):
        st.markdown("""
        **Performance Metrics Explained:**
        - **Accuracy**: Overall correctness of predictions
        - **Precision**: Of predicted churns, how many actually churned (fewer false positives)
        - **Recall**: Of actual churns, how many were caught (fewer false negatives)
        - **F1-Score**: Harmonic mean of Precision and Recall (balanced metric)
        - **ROC-AUC**: Ability to distinguish between churn and non-churn (higher is better, max = 1.0)
        
        **Best Model**: Higher values are better for all metrics. ROC-AUC is often the most important for imbalanced datasets.
        """)
    
    try:
        from src.model_comparison_utils import (
            load_model_comparison_data, 
            enhance_comparison_dataframe,
            get_model_parameters,
            get_feature_importance_comparison,
            create_radar_chart_data
        )
        import io
        
        # Load and enhance comparison data
        @st.cache_data
        def load_comparison_data():
            df = load_model_comparison_data()
            if df is not None:
                return enhance_comparison_dataframe(df)
            return None
        
        df = load_comparison_data()
        
        if df is not None:
            
            # Find best model for each metric
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
            available_metrics = [m for m in metrics if m in df.columns]
            
            # Calculate overall ranking
            df_ranked = df.copy()
            for metric in available_metrics:
                df_ranked[f'{metric}_rank'] = df_ranked[metric].rank(ascending=False, method='min')
            
            # Overall score (average rank)
            rank_cols = [f'{m}_rank' for m in available_metrics]
            df_ranked['Overall_Rank'] = df_ranked[rank_cols].mean(axis=1)
            df_ranked = df_ranked.sort_values('Overall_Rank')
            
            # Display summary metrics
            st.subheader("🏆 Model Rankings")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            metric_best = {}
            for i, metric in enumerate(available_metrics[:5]):
                best_idx = df[metric].idxmax()
                best_model = df.loc[best_idx, 'Model']
                best_value = df.loc[best_idx, metric]
                metric_best[metric] = (best_model, best_value)
                
                with [col1, col2, col3, col4, col5][i]:
                    st.metric(
                        f"Best {metric}",
                        f"{best_value:.4f}",
                        best_model
                    )
            
            # Best overall model
            best_overall = df_ranked.iloc[0]
            st.success(f"🏆 **Best Overall Model**: **{best_overall['Model']}** (Average Rank: {best_overall['Overall_Rank']:.2f})")
            
            st.markdown("---")
            
            # Tabs for different visualizations
            tab_overview, tab_radar, tab_heatmap, tab_details = st.tabs([
                "📊 Overview", "🕸️ Radar Chart", "🔥 Feature Importance", "📋 Details"
            ])
            
            with tab_overview:
                # Bar charts for key metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Accuracy Comparison")
                    fig_acc = go.Figure()
                    fig_acc.add_trace(go.Bar(
                        x=df['Model'],
                        y=df['Accuracy'],
                        text=[f"{val:.3f}" for val in df['Accuracy']],
                        textposition='auto',
                        marker_color='#1f77b4',
                        hovertemplate="<b>%{x}</b><br>Accuracy: %{y:.4f}<extra></extra>"
                    ))
                    fig_acc.update_layout(
                        yaxis_title="Accuracy",
                        yaxis=dict(range=[0, 1]),
                        height=400,
                        template='plotly_dark'
                    )
                    st.plotly_chart(fig_acc, use_container_width=True)
                
                with col2:
                    st.markdown("#### F1-Score Comparison")
                    fig_f1 = go.Figure()
                    fig_f1.add_trace(go.Bar(
                        x=df['Model'],
                        y=df['F1-Score'],
                        text=[f"{val:.3f}" for val in df['F1-Score']],
                        textposition='auto',
                        marker_color='#d62728',
                        hovertemplate="<b>%{x}</b><br>F1-Score: %{y:.4f}<extra></extra>"
                    ))
                    fig_f1.update_layout(
                        yaxis_title="F1-Score",
                        yaxis=dict(range=[0, 1]),
                        height=400,
                        template='plotly_dark'
                    )
                    st.plotly_chart(fig_f1, use_container_width=True)
                
                # ROC-AUC comparison
                st.markdown("#### ROC-AUC Comparison")
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Bar(
                    x=df['Model'],
                    y=df['ROC-AUC'],
                    text=[f"{val:.3f}" for val in df['ROC-AUC']],
                    textposition='auto',
                    marker_color='#9467bd',
                    hovertemplate="<b>%{x}</b><br>ROC-AUC: %{y:.4f}<extra></extra>"
                ))
                fig_roc.update_layout(
                    yaxis_title="ROC-AUC",
                    yaxis=dict(range=[0, 1]),
                    height=400,
                    template='plotly_dark'
                )
                st.plotly_chart(fig_roc, use_container_width=True)
                
                # Full comparison chart
                st.markdown("#### All Metrics Comparison")
                fig = go.Figure()
                
                colors = {
                    'Accuracy': '#1f77b4',
                    'Precision': '#ff7f0e',
                    'Recall': '#2ca02c',
                    'F1-Score': '#d62728',
                    'ROC-AUC': '#9467bd'
                }
                
                for metric in available_metrics:
                    fig.add_trace(go.Bar(
                        name=metric,
                        x=df['Model'],
                        y=df[metric],
                        text=[f"{val:.3f}" for val in df[metric]],
                        textposition='outside',
                        marker_color=colors.get(metric, '#17becf'),
                        hovertemplate=f"<b>%{{x}}</b><br>{metric}: %{{y:.4f}}<extra></extra>"
                    ))
                
                fig.update_layout(
                    title="Model Performance Comparison - All Metrics",
                    xaxis_title="Model",
                    yaxis_title="Score",
                    yaxis=dict(range=[0, 1]),
                    barmode='group',
                    height=600,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    hovermode='x unified',
                    template='plotly_dark'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with tab_radar:
                st.markdown("#### Radar Chart - Multi-Metric Comparison")
                
                # Prepare radar chart data
                metric_names, radar_data = create_radar_chart_data(df)
                
                if metric_names and radar_data:
                    fig_radar = go.Figure()
                    
                    colors_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
                    
                    for idx, (model_name, values) in enumerate(radar_data.items()):
                        # Close the radar chart
                        values_closed = values + [values[0]]
                        metric_names_closed = metric_names + [metric_names[0]]
                        
                        fig_radar.add_trace(go.Scatterpolar(
                            r=values_closed,
                            theta=metric_names_closed,
                            fill='toself',
                            name=model_name,
                            line_color=colors_list[idx % len(colors_list)]
                        ))
                    
                    fig_radar.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 1]
                            )
                        ),
                        showlegend=True,
                        height=600,
                        template='plotly_dark'
                    )
                    
                    st.plotly_chart(fig_radar, use_container_width=True)
            
            with tab_heatmap:
                st.markdown("#### Feature Importance Heatmap")
                
                # Get feature importance comparison
                model_names = [name.lower().replace(' ', '_') for name in df['Model'].tolist()]
                importance_df = get_feature_importance_comparison(model_names)
                
                if not importance_df.empty:
                    # Get top features
                    top_n = st.slider("Show Top N Features", 5, 30, 15, key="importance_top_n")
                    top_features = importance_df.mean(axis=1).nlargest(top_n).index
                    importance_top = importance_df.loc[top_features]
                    
                    # Create heatmap
                    fig_heatmap = go.Figure(data=go.Heatmap(
                        z=importance_top.values,
                        x=importance_top.columns,
                        y=importance_top.index,
                        colorscale='Viridis',
                        text=importance_top.values,
                        texttemplate='%{text:.3f}',
                        textfont={"size": 10},
                        hovertemplate="<b>%{y}</b><br>Model: %{x}<br>Importance: %{z:.4f}<extra></extra>"
                    ))
                    
                    fig_heatmap.update_layout(
                        title="Feature Importance Comparison Across Models",
                        xaxis_title="Model",
                        yaxis_title="Feature",
                        height=max(500, top_n * 30),
                        template='plotly_dark'
                    )
                    
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                else:
                    st.info("Feature importance data not available for all models.")
            
            with tab_details:
                # Detailed comparison table
                st.subheader("📋 Detailed Performance Table")
                
                display_df = df.copy()
                for metric in available_metrics:
                    display_df[metric] = display_df[metric].round(4)
                
                display_df['Rank'] = df_ranked['Overall_Rank'].round(2).values
                display_df = display_df[['Model', 'Rank'] + available_metrics]
                
                if 'Inference_Time_ms' in display_df.columns:
                    display_df['Inference_Time_ms'] = display_df['Inference_Time_ms'].round(2)
                    display_df = display_df[['Model', 'Rank'] + available_metrics + ['Inference_Time_ms']]
                
                display_df = display_df.sort_values('Rank')
                
                st.dataframe(display_df, use_container_width=True, height=400)
                
                # Model parameters
                st.markdown("---")
                st.subheader("🔧 Model Parameters")
                
                selected_model_detail = st.selectbox("Select Model to View Parameters", df['Model'].tolist())
                
                model_name_lower = selected_model_detail.lower().replace(' ', '_')
                params = get_model_parameters(model_name_lower)
                
                if params:
                    with st.expander(f"Parameters for {selected_model_detail}", expanded=False):
                        st.json(params)
                else:
                    st.info(f"Parameters not available for {selected_model_detail}")
                
                # Download section
                st.markdown("---")
                st.subheader("📥 Download")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Download CSV
                    csv_buffer = io.StringIO()
                    display_df.to_csv(csv_buffer, index=False)
                    st.download_button(
                        label="📥 Download Comparison (CSV)",
                        data=csv_buffer.getvalue(),
                        file_name=f"model_comparison_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    # Download model pickle
                    project_root = Path(__file__).parent
                    best_model_name = best_overall['Model'].lower().replace(' ', '_')
                    model_path = project_root / "models" / f"{best_model_name}.pkl"
                    
                    if model_path.exists():
                        with open(model_path, 'rb') as f:
                            model_bytes = f.read()
                        st.download_button(
                            label="📥 Download Best Model (PKL)",
                            data=model_bytes,
                            file_name=f"{best_model_name}_model.pkl",
                            mime="application/octet-stream"
                        )
                    else:
                        st.info("Model file not found")
                
                with col3:
                    st.info("💡 PDF report generation coming soon!")
            
            # Model recommendations
            st.markdown("---")
            st.subheader("💡 Model Recommendations")
            
            best_roc = df.loc[df['ROC-AUC'].idxmax(), 'Model']
            best_precision = df.loc[df['Precision'].idxmax(), 'Model']
            best_recall = df.loc[df['Recall'].idxmax(), 'Model']
            
            if 'Inference_Time_ms' in df.columns:
                fastest = df.loc[df['Inference_Time_ms'].idxmin(), 'Model']
                recommendations = f"""
                **Based on Performance Metrics:**
                
                - 🎯 **Best for Overall Performance (ROC-AUC)**: **{best_roc}** - Best at distinguishing churn vs non-churn
                - 🎯 **Best for Precision**: **{best_precision}** - Fewest false positives
                - 🎯 **Best for Recall**: **{best_recall}** - Catches the most actual churns
                - ⚡ **Fastest Inference**: **{fastest}** - {df.loc[df['Inference_Time_ms'].idxmin(), 'Inference_Time_ms']:.2f} ms per prediction
                
                **Recommendation**: For customer churn prediction, **ROC-AUC** is typically the most important metric.
                """
            else:
                recommendations = f"""
                **Based on Performance Metrics:**
                
                - 🎯 **Best for Overall Performance (ROC-AUC)**: **{best_roc}** - Best at distinguishing churn vs non-churn
                - 🎯 **Best for Precision**: **{best_precision}** - Fewest false positives
                - 🎯 **Best for Recall**: **{best_recall}** - Catches the most actual churns
                
                **Recommendation**: For customer churn prediction, **ROC-AUC** is typically the most important metric.
                """
            st.markdown(recommendations)
            
        else:
            st.warning("⚠️ Model comparison data not found.")
            st.info("💡 Run model training first to generate comparisons: `python src/model_training.py`")
    except Exception as e:
        st.error(f"❌ Error loading model comparison: {e}")
        st.exception(e)


