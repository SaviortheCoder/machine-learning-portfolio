# 03 · Employee Retention Prediction — Full Classification Pipeline

## Problem
Binary classification: will a candidate change jobs? Compares seven classifiers on an imbalanced HR dataset.

## Dataset
**HR Analytics Job Change** (hrdata.csv) — same as Project 01, now used for modeling.

## What This Notebook Does
1. Preprocessing — label encoding, one-hot encoding, imputation
2. SMOTE oversampling — corrects 75/25 class imbalance
3. PCA dimensionality reduction experiment
4. Logistic Regression, KNN (GridSearchCV), Naive Bayes, SVM, Decision Tree, Random Forest, Gradient Boosting
5. Classification report and confusion matrix for each model
6. Model comparison: Accuracy, Precision, Recall, F1, ROC-AUC

## Key Results
- Gradient Boosting and Random Forest: highest AUC (~0.84)
- SMOTE improves minority recall by ~15-20%
- `city_development_index` and `experience` rank as top features

## How to Run
```bash
cd 03-classification-models
pip install imbalanced-learn
jupyter notebook notebook.ipynb
```
