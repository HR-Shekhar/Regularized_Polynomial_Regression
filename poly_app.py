import streamlit as st
import numpy as np
import pickle

# Load both models
with open("PolyFromScratch.pkl", "rb") as f:
    weights = pickle.load(f)

with open("PolySklearn.pkl", "rb") as f:
    sklearn_model = pickle.load(f)

# Scratch model prediction function
def predict_scratch(x, weights):
    x = np.insert(x, 0, 1, axis=1)  # Add bias term
    return x @ weights

# Streamlit UI
st.set_page_config(page_title="Polynomial Ridge Regression", layout="centered")
st.title("🔧 Quality Rating Prediction: Polynomial Ridge Regression")

st.markdown("This app compares predictions of Polynomial Ridge Regression using:")
st.markdown("- 🧠 Custom model (from scratch)")
st.markdown("- 🤖 Scikit-learn pipeline")

st.header("🧪 Input Features")
temperature = st.number_input("Temperature (°C)", value=210.0)
pressure = st.number_input("Pressure (kPa)", value=8.0)
transformation = st.number_input("Material Transformation Metric", value=9_000_000.0)

# Feature engineering
temp_pressure = temperature * pressure

# Prepare input for prediction
x = np.array([[temp_pressure, transformation]])

if st.button("Predict"):
    pred_scratch = predict_scratch(x, weights)[0]
    pred_sklearn = sklearn_model.predict(x)[0]

    st.success(f"🧠 From Scratch Prediction: **{pred_scratch:.2f}**")
    st.success(f"🤖 Scikit-learn Prediction: **{pred_sklearn:.2f}**")

st.markdown("---")
st.caption("Made by Himanshu Shekhar")
