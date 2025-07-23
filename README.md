# Regression Models Dashboard

## Overview
The **Regression Models Dashboard** is an intuitive and interactive web application designed to help users explore and understand different types of regression models. It provides a no-code/low-code interface ideal for both students and practitioners aiming to analyze datasets and gain insights using statistical modeling.

---

## Key Features

- **Model Variety**  
  Supports three types of regression models:
  - Multiple Linear Regression
  - Polynomial Regression
  - Logistic Regression

- **CSV Upload Support**  
  Users can upload custom datasets (CSV format, up to 200MB) to build and analyze models on their own data.
<img width="1842" height="723" alt="Screenshot 2025-06-07 005338" src="https://github.com/user-attachments/assets/0f150621-a849-450f-a7ff-b84e83016f7e" />


- **Dynamic Variable Selection**  
  - Easily select the dependent (target) and independent (predictor) variables from your dataset.
  - Categorical columns (e.g., Product types) are automatically encoded for regression modeling.

- **Polynomial Degree Slider**  
  For polynomial regression, adjust the polynomial degree dynamically using a slider (degree 1 to 5).

- **Model Output & Summary**  
  - For Linear and Polynomial Regression:
    - R-squared and Adjusted R-squared
    - Coefficient values and statistical significance (p-values)
    - Confidence intervals
    - Residual diagnostics and RMSE (Root Mean Squared Error)
  - For Logistic Regression:
    - Classification probabilities
    - Confusion matrix and accuracy
    - ROC curve and AUC (if included in future updates)

- **Prediction Interface**  
  - Input values for predictor variables and receive predicted values.
  - Output includes 95% prediction intervals and standard errors.

- **Export Options**  
  - Download detailed model summaries.
  - Download prediction logs for external analysis.

---
<img width="339" height="306" alt="Screenshot 2025-06-07 005226" src="https://github.com/user-attachments/assets/30b5c9c5-117a-4993-9be5-41067a0a7f1e" />


## Educational Focus

This tool is especially valuable in academic and training settings. It helps users:

- Visually interpret the impact of variable selection and transformations.
- Understand regression diagnostics through real-time feedback.
- Learn by experimenting with real-world or synthetic datasets.
- Observe how polynomial terms influence model fit.
- Explore multicollinearity warnings and how categorical encoding affects model interpretation.

---

## Typical Workflow

1. **Select Regression Type**  
   Choose from Multiple Linear, Polynomial, or Logistic Regression.

2. **Upload Dataset (CSV)**  
   Drag and drop or browse to upload a `.csv` file containing your data.

3. **Variable Configuration**  
   - Choose your dependent variable.
   - Select independent variable(s) from the dataset.
   - For polynomial regression, adjust the degree slider as needed.

4. **Model Generation**  
   Instantly generate a regression model with outputs such as coefficients, errors, R² scores, and plots.

5. **Predict Outcomes**  
   Enter values for your predictors and get immediate model-based predictions with confidence intervals.

6. **Export Results**  
   Download summaries and logs for record-keeping or further analysis.

---

<img width="1817" height="610" alt="Screenshot 2025-06-07 005219" src="https://github.com/user-attachments/assets/654c3b0d-a2fb-4e37-9bc9-bdef0d09f4b3" />
<img width="1814" height="335" alt="Screenshot 2025-06-07 005625" src="https://github.com/user-attachments/assets/9b253a6c-7a06-4537-9561-f0c4fbf3f1c3" />

## Technical Notes

- Input Format: CSV files with headers, ideally clean data (missing values handled externally).
- Data Types: Supports both numerical and categorical data (categorical variables are encoded automatically).
- Limitations:
  - Logistic regression currently supports binary classification only.
  - No automated outlier detection.
  - Time-based models (e.g., time series) are not included.

---

## Example Use Cases

- **Education:** Help students grasp regression fundamentals with hands-on experimentation.
- **Retail Analytics:** Predict purchases based on features like sales, profit, and product type.
- **Real Estate:** Model area vs. price with polynomial regression.
- **Marketing:** Analyze returns vs. campaign spend, duration, and product category.

---

## Notes & Warnings

- If a high **condition number** is shown in the output (e.g., `1.03e+13`), it indicates potential multicollinearity. Consider removing or transforming correlated variables.
- **P-values** < 0.05 usually indicate statistical significance for corresponding predictors.
- **Residual plots and error metrics** help diagnose model fit and outliers.

---

## Future Enhancements

- Visual residual plots and correlation heatmaps
- Support for regularized regression (Ridge, Lasso)
- Auto feature scaling and transformation
- Support for time-series regression
- Advanced export formats (e.g., PDF reports)
