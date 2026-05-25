# Machine Learning Portfolio

CAP 4611 – Algorithms for Machine Learning · Spring 2026

---

## About

This repository contains five end-to-end machine learning projects covering the full ML pipeline: raw data ingestion, preprocessing, feature engineering, model training, evaluation, and interpretation.

These projects directly feed into larger independent work, including [Project Blackrock](https://github.com/SaviortheCoder), an ML-based swing trading system targeting large-cap US equities.

---

## Projects

| # | Project | Techniques | Dataset |
|---|---------|------------|---------|
| 1 | [Candidate EDA](#01--candidate-job-change-eda) | EDA, Missing Data, Distributions, Correlations | HR Analytics |
| 2 | [Customer Spending Prediction](#02--customer-spending-prediction) | Linear Regression, Normal Equation, Gradient Descent, Regularization | E-Commerce |
| 3 | [Employee Retention Prediction](#03--employee-retention-prediction) | Logistic Regression, KNN, Naive Bayes, SVM, Decision Trees, Random Forest, Boosting | HR Analytics |
| 4 | [Customer Segmentation](#04--customer-segmentation-clustering) | KMeans, Elbow Method, Agglomerative Clustering | HR Analytics |
| 5 | [Fake News Classification (NLP)](#05--fake-news-classification-nlp--deep-learning) | TF-IDF, Multinomial Naive Bayes, MLP Neural Network | ISOT Fake News |

---

## 01 · Candidate Job Change EDA

**Problem:** A company wants to understand patterns in candidates who are likely to leave after training — useful for HR planning and reducing onboarding waste.

**Approach:**
- Identified missing data patterns across categorical and numerical features
- Analyzed distributions, outliers, and class imbalance in the target variable
- Computed correlation heatmaps and pairplots to surface feature relationships
- Generated actionable feature engineering recommendations

**Key Findings:**
- ~25% of candidates had missing `experience` and `company_size` values — both correlated with job change likelihood
- Strong class imbalance (~75/25 stay/leave split) noted for future modeling considerations
- City development index and years of experience showed the clearest separation between classes

📓 [View Notebook](01-eda-analysis/hr-candidate-eda.ipynb)

---

## 02 · Customer Spending Prediction

**Problem:** Predict annual customer spending on an e-commerce platform based on behavioral features (session time, time on app, membership length).

**Approach:**
- Implemented Linear Regression from scratch using the **Normal Equation** and compared to scikit-learn's solution
- Built **Batch Gradient Descent** and **Stochastic Gradient Descent** manually, confirming convergence behavior
- Applied **Ridge**, **Lasso**, and **Elastic Net** regularization and analyzed their effect on coefficients
- Evaluated with MAE, RMSE, and R²

**Highlights:**
- Manual gradient descent matched sklearn's coefficients within numerical tolerance
- Lasso drove less-informative features to exactly zero — natural feature selection
- R² of ~0.98 on test set — the features are highly predictive

📓 [View Notebook](02-regression-models/customer-spending-regression.ipynb)

---

## 03 · Employee Retention Prediction

**Problem:** Binary classification — will a candidate change jobs? Full model comparison across seven classifiers.

**Approach:**
- Full preprocessing pipeline: label encoding, one-hot encoding, SMOTE oversampling for class imbalance
- PCA dimensionality reduction explored alongside full-feature training
- Grid search hyperparameter tuning for KNN
- Models compared: Logistic Regression, KNN, Naive Bayes, SVM, Decision Tree, Random Forest, Gradient Boosting
- Evaluated with Precision, Recall, F1, ROC-AUC

**Highlights:**
- SMOTE significantly improved recall on the minority class across all models
- Random Forest and Gradient Boosting outperformed simpler classifiers on AUC
- Decision Tree without pruning showed clear overfitting vs. ensemble methods

📓 [View Notebook](03-classification-models/employee-retention-classification.ipynb)

---

## 04 · Customer Segmentation (Clustering)

**Problem:** Unsupervised segmentation of HR candidates into meaningful groups without labels.

**Approach:**
- Applied **KMeans** with Elbow Method to determine optimal cluster count
- Applied **Agglomerative Clustering** (Ward linkage) and compared segment composition
- Visualized clusters in 2D using PCA projection
- Profiled each cluster to generate interpretable business personas

**Highlights:**
- Elbow method identified k=4 as the inflection point
- Agglomerative clustering produced more balanced segments vs. KMeans
- Cluster 2 ("experienced city professionals") showed the highest job-change probability — directly actionable for HR

📓 [View Notebook](04-clustering/customer-segmentation-clustering.ipynb)

---

## 05 · Fake News Classification (NLP + Deep Learning)

**Problem:** Binary classification of news articles as fake or real using NLP pipelines and neural networks, on the ISOT Fake News dataset (~45k articles).

**Approach:**
- Text preprocessing: lowercasing, punctuation removal, stopword removal, lemmatization
- Feature extraction: CountVectorizer → TF-IDF transformation (no data leakage — subject column excluded)
- Model 1: **Multinomial Naive Bayes** pipeline — fast baseline (~90-94% accuracy)
- Model 2: **MLP Neural Network** (2 hidden layers, sklearn MLPClassifier) — high-accuracy classifier (~99%)
- Extra Credit: Subject multi-class classifier achieving 93% across 8 categories

**Key Design Decision:**
The `subject` column was intentionally excluded from the feature text. Subject values directly encode the label in this dataset (e.g., `politicsNews` = real, `leftNews` = fake), so including it would inflate accuracy through data leakage rather than genuine learned signal.

📓 [View Notebook](05-nlp-deep-learning/fake-news-nlp-classification.ipynb)

---

## Setup

```bash
# Clone the repo
git clone https://github.com/SaviortheCoder/machine-learning-portfolio.git
cd machine-learning-portfolio

# Create environment (Python 3.11)
conda create -n ml_portfolio python=3.11
conda activate ml_portfolio

# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn nltk wordcloud imbalanced-learn missingno scipy

# Download NLTK data (required for NLP notebook)
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"
```

Each project folder contains its own README with dataset download instructions.

---

## Tech Stack

`Python 3.11` · `scikit-learn` · `pandas` · `numpy` · `matplotlib` · `seaborn` · `NLTK` · `imbalanced-learn` · `Jupyter`
