import streamlit as st
import pandas as pd
import joblib
import os
import pickle
import plotly.express as px
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix
import numpy as np

st.set_page_config(page_title="Model Performance (Beta)", page_icon="🧪", layout="wide")

st.title("🧪 Model Performance Evaluation (Experimental)")
st.warning("""
**⚠️ BETA FEATURE**: This model is currently in **experimental mode**.
- **High False Positive Rate**: The model is tuned for high recall (catching fraud) but flags many legitimate transactions as suspicious.
- **Do not use for automated blocking**: Use only for risk scoring and investigation.
""")

@st.cache_resource
def load_models():
    models = {}
    # Debug paths
    base_dir = "models"
    if not os.path.exists(base_dir):
        if os.path.exists("../models"):
            base_dir = "../models"
        else:
            st.error(f"Could not find models directory. CWD: {os.getcwd()}")
            st.write(f"Files in CWD: {os.listdir('.')}")
            return {}

    try:
        # Load RF Pipeline
        rf_path = os.path.join(base_dir, "fraud_detection_rf_pipeline.pkl")
        if os.path.exists(rf_path):
            models['Random Forest'] = joblib.load(rf_path)
        
        # Load XGBoost Pipeline (formerly thought to be LR)
        xgb_path = os.path.join(base_dir, "fraud_detection_pipeline.pkl")
        if os.path.exists(xgb_path):
            models['XGBoost'] = joblib.load(xgb_path)
            
        # Load Threshold
        thresh_path = os.path.join(base_dir, "fraud_threshold.pkl")
        if os.path.exists(thresh_path):
            models['threshold'] = joblib.load(thresh_path)
        else:
            models['threshold'] = 0.5
            
    except Exception as e:
        st.error(f"Error loading models: {e}")
    
    return models

# ... (data loading)

models = load_models()

# Add Model Inspection to Sidebar
with st.sidebar:
    with st.expander("🛠️ Model Debug Info"):
        st.write("Loaded Models:")
        for name, model in models.items():
            if name != 'threshold':
                st.write(f"**{name}**")
                try:
                    # Show the final estimator type
                    st.code(str(model.steps[-1][1]))
                except:
                    st.write("Could not retrieve estimator details")

import utils

# ... (Config)

# ... (load_models defined above)

# Use shared data loader
with st.spinner("Loading test data... (Shared Cache)"):
    df, col_map = utils.load_shared_data()
    
# Sample Test Data logic refactored
if not df.empty:
    target_col = col_map['fraud']
    
    if target_col:
        # Sample 20k rows for speed
        df_sample = df.sample(n=20000, random_state=42)
        X_test = df_sample.drop(columns=[target_col])
        y_test = df_sample[target_col]
    else:
        X_test, y_test = None, None
else:
    X_test, y_test = None, None

# ... (Analysis continues)

if not models:
    st.warning("No models found in 'models/' directory.")
elif X_test is None:
    st.warning("Could not load test data.")
else:
    st.success(f"Loaded {len(models)-1} models. Evaluated on {len(X_test)} samples.")
    
    tabs = st.tabs(["Performance Metrics", "ROC Curves", "Confusion Matrix"])
    
    # Run Predictions
    results = {}
    for name, model in models.items():
        if name == 'threshold': continue
        try:
            prob = model.predict_proba(X_test)[:, 1]
            results[name] = prob
        except Exception as e:
             st.error(f"Failed to predict with {name}: {e}")

    with tabs[0]: # Metrics
        st.subheader("Key Metrics")
        
        for name, prob in results.items():
            st.markdown(f"### {name}")
            
            # Use optimal threshold for RF if available
            thresh = models.get('threshold', 0.5) if 'Random' in name else 0.5
            pred = (prob >= thresh).astype(int)
            
            # Handle cases where model predicts only one class in sample
            try:
                report = classification_report(y_test, pred, output_dict=True)
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Accuracy", f"{report['accuracy']:.4f}")
                # Check if '1' (Fraud) exists in report
                if '1' in report:
                    c2.metric("Precision (Fraud)", f"{report['1']['precision']:.4f}")
                    c3.metric("Recall (Fraud)", f"{report['1']['recall']:.4f}")
                    c4.metric("F1-Score", f"{report['1']['f1-score']:.4f}")
                else:
                    st.warning("No fraud cases predicted in this sample.")
            except Exception as e:
                st.error(f"Error calculating metrics: {e}")
            st.divider()

    with tabs[1]: # ROC
        st.subheader("ROC Curve Comparison")
        fig_roc = px.area(title="ROC Curve", labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'})
        
        for name, prob in results.items():
            try:
                fpr, tpr, _ = roc_curve(y_test, prob)
                auc_score = auc(fpr, tpr)
                fig_roc.add_scatter(x=fpr, y=tpr, name=f"{name} (AUC={auc_score:.4f})", mode='lines')
            except:
                pass
            
        # fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1) (Removed as per request)
        st.plotly_chart(fig_roc)

    with tabs[2]: # Confusion Matrix
        col_cm1, col_cm2 = st.columns(2)
        idx = 0
        cols = [col_cm1, col_cm2]
        
        for name, prob in results.items():
            with cols[idx % 2]:
                st.write(f"**{name}**")
                thresh = models.get('threshold', 0.5) if 'Random' in name else 0.5
                pred = (prob >= thresh).astype(int)
                
                try:
                    cm = confusion_matrix(y_test, pred)
                    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                                       labels=dict(x="Predicted Class (0=Normal, 1=Fraud)", 
                                                   y="Actual Class (0=Normal, 1=Fraud)", 
                                                   color="Count"),
                                       x=['Normal', 'Fraud'], y=['Normal', 'Fraud'])
                    st.plotly_chart(fig_cm)
                except:
                    st.write("Could not generate Confusion Matrix for this sample.")
            idx += 1
