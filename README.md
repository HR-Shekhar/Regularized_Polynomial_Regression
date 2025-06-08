
# 🔧 Polynomial Regression: From Scratch vs Scikit-learn

This project demonstrates how to build and compare **Polynomial Regression** models using:

- ✅ From Scratch (NumPy + Gradient Descent)
- ✅ Scikit-learn (`Pipeline`, `PolynomialFeatures`, `Ridge`)

The model predicts a **Quality Rating** for materials based on engineered features derived from temperature, pressure, and transformation metrics.

---

## 📊 Problem Statement

Given the engineered features:

- `Temperature x Pressure`
- `Material Transformation Metric`

Predict the **Quality Rating** of a material.

---

## 🛠 Features

| Model | Library | Notes |
|-------|---------|-------|
| ✅ From Scratch | NumPy | Manual Polynomial Transformation + L2 Ridge Regularization |
| ✅ Scikit-learn | `Pipeline` + `PolynomialFeatures` + `Ridge` | Scaled and trained model |

---

## 🚀 Try the Web App

Live comparison of both models on Streamlit:

🔗 [Launch Polynomial Regression App](https://your-poly-app-url.streamlit.app)  
_(Replace with your actual deployed link)_

---

## 📁 Files Included

- `Polynomomial_Regression.ipynb`: Notebook training both models
- `app_poly.py`: Streamlit app to compare the predictions
- `poly_ridge_scratch.pkl`: Pickle file with weights, poly, scaler
- `poly_ridge_sklearn.pkl`: Trained sklearn pipeline
- `requirements.txt`: Dependencies to run the app

---

## 💻 Run Locally

```bash
pip install -r requirements.txt
streamlit run app_poly.py
```

---

## 🧠 Learning Outcomes

- Learn how to transform data for polynomial regression
- Understand overfitting and regularization
- Compare performance of raw NumPy implementation vs sklearn pipelines
- Deploy ML models online using Streamlit

---

## 👨‍💻 Author

Made with ❤️ by **Himanshu Shekhar** — building and deploying real ML models from scratch and understanding the fundamentals of machine learning.

---

⭐️ Star this repo if you find it helpful!
