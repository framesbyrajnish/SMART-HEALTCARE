import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

st.set_page_config(page_title='Smart Healthcare — Risk Predictor')
BASE = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE / 'models'

st.title('Smart Healthcare — Risk Predictor')
model_choice = st.sidebar.selectbox('Model', ['Diabetes', 'Heart'])

if model_choice == 'Diabetes':
    model_path = MODELS_DIR / 'diabetes_rf.joblib'
    scaler_path = MODELS_DIR / 'pima_scaler.joblib'
else:
    model_path = MODELS_DIR / 'heart_rf.joblib'
    scaler_path = MODELS_DIR / 'heart_scaler.joblib'

if not model_path.exists():
    st.error('Model not found. Train models with scripts/train_all.py')

with st.form('input_form'):
    st.header('Patient features')
    pregnancies = st.number_input('Pregnancies', 0, 20, 1)
    glucose = st.number_input('Glucose', 0, 300, 120)
    bp = st.number_input('BloodPressure', 0, 200, 70)
    bmi = st.number_input('BMI', 0.0, 70.0, 30.0)
    age = st.number_input('Age', 0, 120, 33)
    submitted = st.form_submit_button('Predict')
    if submitted:
        st.write('Prediction flow will run here—ensure models are trained locally')
