import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load models
scratch_model = joblib.load("PolyFromScratch.pkl")
sklearn_model = joblib.load("PolySklearn.pkl")

# Streamlit UI
st.title("🧪 Model Comparison: Scratch vs Scikit-learn")
st.subtitle("This model predicts the product quality on the basis of features like Temperature, Pressure and Material Transformation Metrics")
st.markdown("Compare predictions from a manually built polynomial regression model and a scikit-learn model.")

# Input fields
pressure = st.number_input("Pressure (in bar)", min_value=0.0, step=0.1)
temperature = st.number_input("Temperature (in °C)", min_value=0.0, step=0.1)
material_metric = st.number_input("Material Transformation Metrics", min_value=0.0, step=0.1)

# Predict button
if st.button("Compare Models"):
    # Original input for sklearn model
    x_raw = np.array([[pressure * temperature, material_metric]])

    # Scratch model input (manually computed feature)
    x_scratch_input = np.array([[pressure * temperature, material_metric]])
    poly_scratch = scratch_model['poly']
    scaler_scratch = scratch_model['scaler']
    w = scratch_model['w']
    b = scratch_model['b']

    x_scaled_scratch = scaler_scratch.transform(x_scratch_input)
    x_poly = poly_scratch.transform(x_scaled_scratch)
    pred_scratch = np.dot(x_poly, w) + b

    # Sklearn model prediction
    pred_sklearn = sklearn_model.predict(x_poly)

    # Prepare comparison dataframe
    df_compare = pd.DataFrame({
        "Model": ["From Scratch", "Scikit-learn"],
        "Predicted Quality Rating": [pred_scratch[0], pred_sklearn[0]]
    })
    df_compare["Predicted Quality Rating"] = df_compare["Predicted Quality Rating"].round(2)

    # Show table
    st.subheader("📊 Prediction Comparison")
    st.table(df_compare)

    # Show difference
    diff = abs(pred_scratch[0] - pred_sklearn[0])
    st.write(f"🧮 Absolute Difference: **{diff:.2f}**")

    # Optional: Plotting bar chart
    fig, ax = plt.subplots()
    ax.bar(df_compare["Model"], df_compare["Predicted Quality Rating"], color=["#3498db", "#2ecc71"])
    ax.set_ylabel("Quality Rating")
    ax.set_title("Model Prediction Comparison")
    st.pyplot(fig)