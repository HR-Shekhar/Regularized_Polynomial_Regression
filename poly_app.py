import streamlit as st
import numpy as np
import joblib

# Load models
scratch_model = joblib.load("PolyFromScratch.pkl")
sklearn_model = joblib.load("PolySklearn.pkl")

# Streamlit UI
st.title("Poly Regression Quality Predictor")
st.markdown("Enter manufacturing conditions to predict quality rating.")

# Input fields
pressure = st.number_input("Pressure (in bar)", min_value=0.0, step=0.1)
temperature = st.number_input("Temperature (in °C)", min_value=0.0, step=0.1)
material_metric = st.number_input("Material Transformation Metric", min_value=0.0, step=0.1)

# Predict button
if st.button("Predict Quality Rating"):
    x_input = np.array([[temperature * pressure, material_metric]])

    # Prediction using scratch model
    poly_scratch = scratch_model['poly']
    scaler_scratch = scratch_model['scaler']
    w = scratch_model['w']
    b = scratch_model['b']

    x_poly_scratch = poly_scratch.transform(x_input)
    x_scaled_scratch = scaler_scratch.transform(x_poly_scratch)
    pred_scratch = np.dot(x_scaled_scratch, w) + b

    # Prediction using sklearn model
    pred_sklearn = sklearn_model.predict(x_input)

    # Display results
    st.subheader("Predicted Quality Ratings")
    st.write(f"🔧 From Scratch Model: **{pred_scratch[0]:.2f}**")
    st.write(f"🤖 Scikit-learn Model: **{pred_sklearn[0]:.2f}**")
