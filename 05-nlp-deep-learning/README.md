# 05 · Fake News Classification — NLP + Deep Learning

## Problem
Binary classification of ~45,000 news articles as Fake (0) or Real (1) using TF-IDF pipelines and two ML models.

## Dataset
**ISOT Fake News Dataset** (news.csv) — comma-separated, columns: `subject`, `title`, `text`, `target`, `date`.
Download: [ISOT Lab](https://www.uvic.ca/engineering/ece/isot/datasets/fake-news/index.php) or Kaggle.

## What This Notebook Does
**Section 1 — EDA**
- Raw file inspection (comma-separated)
- Null removal, subject distribution (8 classes), fake/real class balance
- Word clouds by class, AllText feature engineering, length analysis, text preprocessing

**Section 2 — Train/Test Split**
- 80/20 stratified split, class distribution verified

**Section 3 — Multinomial Naive Bayes**
- Pipeline: CountVectorizer → TfidfTransformer → MultinomialNB
- Classification report, confusion matrix, real-world article prediction

**Section 4 — MLP Neural Network**
- Pipeline: CountVectorizer → TfidfTransformer → MLPClassifier (2 hidden layers)
- Classification report, confusion matrix, model comparison

**Extra Credit — Subject Classifier**
- 8-class MLP subject classifier, ~93% accuracy

## Why Subject Is Excluded from AllText
`subject` directly encodes the label in this dataset — including it would cause data leakage and inflate accuracy artificially. AllText is built as `title + text` only.

## Results
| Model | Accuracy |
|-------|----------|
| Multinomial Naive Bayes | ~90-94% |
| MLP Neural Network | ~99% |
| Subject Classifier (EC) | ~93% |

## How to Run
```bash
cd 05-nlp-deep-learning
# Place news.csv in this folder
pip install nltk wordcloud
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"
jupyter notebook notebook.ipynb
```
