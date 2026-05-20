# 02 · Customer Spending Prediction — Regression Pipeline

## Problem
Predict annual customer spending on an e-commerce platform from behavioral features: session length, time on app, time on website, and membership length.

## Dataset
**E-Commerce Customers** (ecommarce.csv) — ~500 customers.
Download: [Kaggle E-Commerce Customers](https://www.kaggle.com/srolka/ecommerce-customers)

## What This Notebook Does
1. EDA — distributions, correlation heatmap, feature selection
2. Train/test split and StandardScaler
3. sklearn LinearRegression — baseline and residual plots
4. Normal Equation — manual implementation, theta comparison
5. Batch Gradient Descent — manual, loss curve
6. Stochastic Gradient Descent — manual, convergence
7. sklearn SGDRegressor — compare to manual SGD
8. Ridge / Lasso / Elastic Net — regularization analysis
9. Polynomial Regression (degree=2)
10. Full metrics table: MAE, RMSE, R²

## Key Results
- Normal Equation and sklearn LR produce identical coefficients
- Lasso drives `Time on Website` to ~0 — natural feature deselection
- Best R² ≈ 0.98; membership length is the dominant predictor

## How to Run
```bash
cd 02-regression-models
jupyter notebook notebook.ipynb
```
