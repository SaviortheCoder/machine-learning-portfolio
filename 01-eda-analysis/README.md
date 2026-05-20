# 01 · Candidate Job Change — Exploratory Data Analysis

## Problem
An HR tech company wants to identify candidates likely to leave after training, enabling smarter onboarding resource allocation.

## Dataset
**HR Analytics Job Change** (hrdata.csv) — ~20,000 candidates.
Download: [Kaggle HR Analytics](https://www.kaggle.com/arashnic/hr-analytics-job-change-of-data-scientists)

## What This Notebook Does
1. Shape, dtype, and sample row inspection
2. Missing value analysis — counts, percentages, missingno matrix
3. Target distribution — class balance analysis
4. Numerical and categorical feature distributions
5. Correlation heatmap — multicollinearity detection
6. Feature insights — variables that separate the two classes

## Key Results
- ~25% missing in `experience` and `company_size` — both predictive
- ~75/25 class imbalance → SMOTE recommended for downstream modeling
- City development index and experience are the top separating features

## How to Run
```bash
cd 01-eda-analysis
jupyter notebook notebook.ipynb
```
