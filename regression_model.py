import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

def prepare_data(df, dep_var, indep_vars, model_type="Multiple Linear", degree=2):
    X = df[indep_vars].copy()
    if 'Month' in X.columns:
        X['Month'] = pd.to_datetime(X['Month'], errors='coerce').dt.month
    X = pd.get_dummies(X, drop_first=True)
    y = pd.to_numeric(df[dep_var], errors='coerce')
    valid_rows = X.notnull().all(axis=1) & y.notnull()
    X = X[valid_rows]
    y = y[valid_rows]

    if model_type == "Polynomial":
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X = poly.fit_transform(X)
        X = pd.DataFrame(X)
        X.insert(0, 'const', 1.0)
        y = y.reset_index(drop=True)
    elif model_type == "Multiple Linear":
        X = sm.add_constant(X)
        X = X.astype('float64')
        y = y.astype('float64')
    elif model_type == "Logistic":
        X = sm.add_constant(X)
        X = X.astype('float64')
        y = y.astype('int')
    return X, y

def run_regression(X, y):
    return sm.OLS(y, X).fit()

def run_polynomial_regression(X, y, degree):
    return sm.OLS(y, X).fit()

def run_logistic_regression(X, y):
    return sm.Logit(y, X).fit()