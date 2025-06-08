import streamlit as st
import numpy as np
from joblib import load

# Load models
ridge_model = load("PolyFromScratch.joblib")   # from-scratch model
sklearn_model = load("PolySklearn.joblib")     # sklearn pipeline

# Prediction function for from-scratch model
def predict_scratch(X, model):
    X_scaled = model["scaler"].transform(X)
    w = model["w"]
    b = model["b"]
    return np.dot(X_scaled, w) + b

# Streamlit UI
st.set_page_config(page_title="Quality Rating Predictor", layout="centered")

st.title("🔧 Quality Rating Prediction: Polynomial Ridge Regression")

st.markdown("""
Compare two models trained on:
- 🧠 **Custom implementation** (from scratch)
- 🤖 **Scikit-learn Ridge regression**
""")

# Inputs
temp = st.number_input("🌡️ Temperature (°C)", value=200.0, step=0.1)
pressure = st.number_input("⏱️ Pressure (kPa)", value=8.0, step=0.01)

if st.button("Predict Quality Rating"):
    # Derived features
    temp_x_pressure = temp * pressure
    mat_trans = temp_x_pressure ** 2

    # Create input for prediction
    X_input = np.array([[temp_x_pressure, mat_trans]])

    # Predict
    pred_scratch = predict_scratch(X_input, ridge_model)
    pred_sklearn = sklearn_model.predict(X_input)[0]

    # Output
    st.subheader("📈 Predictions")
    st.success(f"🧠 From-Scratch Model: **{pred_scratch:.4f}**")
    st.success(f"🤖 Scikit-learn Model: **{pred_sklearn:.4f}**")
