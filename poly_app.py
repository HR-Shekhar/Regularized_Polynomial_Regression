
import streamlit as st
import numpy as np
import joblib

# Load saved models
scratch_model = joblib.load("PolyFromScratch.pkl")  # Dict with 'w', 'b', 'scaler', 'poly'
sklearn_model = joblib.load("PolySklearn.pkl")  # Trained sklearn pipeline

# Unpack scratch model components
w = scratch_model["w"]
b = scratch_model["b"]
scaler = scratch_model["scaler"]
poly = scratch_model["poly"]

# App configuration
st.set_page_config(page_title="Compare Polynomial Ridge Models", page_icon="🔧", layout="centered")
st.title("🔧 Quality Rating Prediction: Polynomial Ridge Regression")

st.markdown("""
This app compares predictions of **Polynomial Ridge Regression** using:
- 🧠 Custom model (from scratch)
- 🤖 Scikit-learn pipeline

**Input features:**
- Temperature × Pressure
- Material Transformation Metric
""")

# Inputs
temp_x_press = st.number_input("Temperature × Pressure", min_value=0.0, step=1.0, value=1500.0)
transformation_metric = st.number_input("Material Transformation Metric", min_value=0.0, step=1000.0, value=9000000.0)

# Predict on button click
if st.button("Predict Quality Rating"):
    x = np.array([[temp_x_press, transformation_metric]])

    # Apply transformation for scratch model
    x_scaled = scaler.transform(x)
    x_poly = poly.transform(x_scaled)
    pred_scratch = np.dot(x_poly, w) + b

    # Predict using sklearn model
    pred_sklearn = sklearn_model.predict(x)[0]

    # Display predictions
    st.subheader("📈 Prediction Comparison")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Scratch Polynomial Ridge", f"{pred_scratch[0]:.2f}")
    with col2:
        st.metric("Scikit-learn Polynomial Ridge", f"{pred_sklearn:.2f}")

st.markdown("---")
st.markdown("Built by **Himanshu Shekhar** | Polynomial Regression Comparison 🔧")
