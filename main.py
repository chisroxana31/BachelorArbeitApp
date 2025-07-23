import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_curve, auc

from data_loader import load_and_prepare
from regression_model import (
    prepare_data,
    run_regression,
    run_polynomial_regression,
    run_logistic_regression
)
from diagnostics import display_diagnostics
from prediction import prediction_interface, forecast_next_month

st.set_page_config(layout="wide")
st.title("Regression Models Dashboard")
st.markdown("Upload your dataset and select a regression type to analyze relationships.")

model_type = st.selectbox("Select regression type:", ["Multiple Linear", "Polynomial", "Logistic"])
uploaded_file = st.file_uploader("Upload your dataset (CSV format only):", type=["csv"])

if uploaded_file:
    df = load_and_prepare(uploaded_file)
    st.subheader("Preview of Dataset")
    st.dataframe(df.head())

    columns = df.columns.tolist()
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_columns:
        st.error("No numeric columns found in the uploaded dataset. Please check your file.")
        st.stop()

    if model_type in ["Multiple Linear", "Polynomial"]:
        dep_var = st.selectbox("Select the dependent variable:", numeric_columns)
    else:
        dep_var = st.selectbox("Select the dependent variable:", columns)

    if model_type == "Logistic":
        st.info("Note: Logistic Regression requires a binary target (0/1).")
        if sorted(df[dep_var].dropna().unique()) not in [[0, 1], [0], [1]]:
            st.error("Selected target is not binary. Choose a different target for logistic regression.")
            st.stop()

    indep_vars = st.multiselect("Select independent variable(s):", [col for col in columns if col != dep_var])

    if dep_var and indep_vars:
        degree = 2
        if model_type == "Polynomial":
            degree = st.slider("Select degree of polynomial:", min_value=2, max_value=5, value=2)

        X, y = prepare_data(df, dep_var, indep_vars, model_type=model_type, degree=degree)

        st.subheader("Debug: Check data types")
        st.write("X dtypes:")
        st.write(X.dtypes)
        st.write("y dtype:")
        st.write(y.dtype)
        st.write("Any non-numeric X?", any(X.dtypes == 'object'))
        st.write("Any missing values?", X.isnull().any().any() or y.isnull().any())

        if model_type == "Multiple Linear":
            model = run_regression(X, y)
        elif model_type == "Polynomial":
            model = run_polynomial_regression(X, y, degree)
        elif model_type == "Logistic":
            model = run_logistic_regression(X, y)

        st.subheader("Regression Summary")
        st.text(model.summary())
        summary_text = model.summary().as_text()
        st.download_button("Download Model Summary", summary_text, file_name="model_summary.txt")

        if model_type in ["Multiple Linear", "Polynomial"]:
            predictions = model.fittedvalues
            rmse = np.sqrt(np.mean((y - predictions) ** 2))
            st.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f}")

        elif model_type == "Logistic":
            preds_binary = (model.predict(X) > 0.5).astype(int)
            acc = accuracy_score(y, preds_binary)
            st.metric("Accuracy", f"{acc:.2%}")

            # Plot ROC Curve
            probs = model.predict(X)
            fpr, tpr, _ = roc_curve(y, probs)
            roc_auc = auc(fpr, tpr)

            fig, ax = plt.subplots(figsize=(4, 2.5))
            ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curve")
            ax.legend()
            st.pyplot(fig, use_container_width=False)

        display_diagnostics(df, X, model, indep_vars)

        if model_type != "Logistic":
            prediction_interface(X, model)
            forecast_next_month(X, model)
else:
    st.info("Awaiting CSV file upload.")
