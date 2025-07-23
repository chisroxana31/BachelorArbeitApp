import streamlit as st
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

def prediction_interface(X, model):
    st.subheader("Make a Prediction")
    input_data = {}
    for var in X.columns:
        if var != 'const':
            input_data[var] = st.number_input(f"Input value for {var}", value=float(X[var].mean()))

    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])
        input_df = sm.add_constant(input_df, has_constant='add')
        prediction = model.get_prediction(input_df)
        pred_summary = prediction.summary_frame(alpha=0.05)
        st.write("Predicted value with 95% prediction interval:")
        st.dataframe(pred_summary)

        # Save input/output log
        log_entry = input_df.copy()
        log_entry["Predicted_Mean"] = pred_summary["mean"].values[0]
        log_df = pd.DataFrame(log_entry)
        st.download_button("Download Prediction Log", log_df.to_csv(index=False), file_name="prediction_log.csv")

def forecast_next_month(X, model):
    st.subheader("Forecast Future Sales")
    if "Month" not in X.columns:
        st.info("Forecasting requires 'Month' as one of the independent variables.")
        return

    last_month = X["Month"].max()
    next_month = last_month + 1

    input_data = X.drop(columns=["const", "Month"]).mean().to_dict()
    input_data["Month"] = next_month

    input_df = pd.DataFrame([input_data])
    input_df = sm.add_constant(input_df, has_constant='add')

    forecast = model.get_prediction(input_df)
    forecast_df = forecast.summary_frame(alpha=0.05)

    st.write(f"Forecast for Month {next_month}")
    st.dataframe(forecast_df)

    # Plot forecasted mean with CI
    fig, ax = plt.subplots(figsize=(4, 2.5))
    ax.errorbar(x=[next_month], y=forecast_df["mean"],
                yerr=[forecast_df["mean_ci_upper"] - forecast_df["mean"]],
                fmt='o', color='blue', label="Forecasted Mean")
    ax.set_title(f"Forecast for Month {next_month}")
    ax.set_xlabel("Month")
    ax.set_ylabel("Sales")
    ax.legend()
    st.pyplot(fig, use_container_width=False)