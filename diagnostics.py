import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

def display_diagnostics(df, X, model, indep_vars):
    st.subheader("Variance Inflation Factors (VIF)")
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    st.dataframe(vif_data)

    st.subheader("Correlation Matrix")
    numeric_vars = df[indep_vars].select_dtypes(include=[np.number])
    if not numeric_vars.empty:
        corr = numeric_vars.corr()
        fig_corr, ax_corr = plt.subplots(figsize=(4, 2.2), dpi=80, tight_layout=True)
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax_corr)
        ax_corr.set_title("Correlation between Numeric Predictors", fontsize=10)
        fig_corr.tight_layout(pad=1.0)
        st.pyplot(fig_corr, use_container_width=False)
    else:
        st.warning("No numeric independent variables available to compute correlation matrix.")

    st.subheader("Residual Analysis")
    if hasattr(model, "resid"):
        residuals = model.resid
        fitted = model.fittedvalues
    else:
        st.warning("Residual analysis is not available for logistic regression.")
        return

    fig1, ax1 = plt.subplots(figsize=(4, 2.2), dpi=80, tight_layout=True)
    sns.residplot(x=fitted, y=residuals, lowess=True, ax=ax1, line_kws={'color': 'red'})
    ax1.set_title("Residuals vs Fitted", fontsize=10)
    fig1.tight_layout(pad=1.0)
    st.pyplot(fig1, use_container_width=False)


    fig2, ax2 = plt.subplots(figsize=(4, 2.2), dpi=80, tight_layout=True)
    sm.qqplot(residuals, line='s', ax=ax2)
    ax2.set_title("Q-Q Plot", fontsize=10)
    fig2.tight_layout(pad=1.0)
    st.pyplot(fig2, use_container_width=False)

    fig3, ax3 = plt.subplots(figsize=(4, 2.2), dpi=80, tight_layout=True)
    sns.histplot(residuals, kde=True, ax=ax3)
    ax3.set_title("Histogram of Residuals", fontsize=10)
    fig3.tight_layout(pad=1.0)
    st.pyplot(fig3, use_container_width=False)

    st.subheader("Influence Diagnostics (Cook’s Distance)")
    influence = model.get_influence()
    (c, _) = influence.cooks_distance
    fig4, ax4 = plt.subplots(figsize=(4, 2.2), dpi=80, tight_layout=True)
    ax4.stem(np.arange(len(c)), c, markerfmt=",")
    ax4.set_title("Cook’s Distance", fontsize=10)
    fig4.tight_layout(pad=1.0)
    st.pyplot(fig4, use_container_width=False)
