# 06 — Image Classification: ANN vs CNN on CIFAR-10

Part of [machine-learning-portfolio](https://github.com/SaviortheCoder/machine-learning-portfolio)

## Overview
Builds and compares two neural network architectures on the CIFAR-10 dataset — 10 classes of 32×32 colour images (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).

| Model | Architecture | Trainable Params | Test Accuracy | Test Loss |
|-------|-------------|-----------------|---------------|-----------|
| ANN   | Dense(1000) → Dense(100) → Dense(50) → Dense(10) | 3,178,660 | **48.88%** | 1.4233 |
| CNN   | Conv2D(32) → Pool → Conv2D(64) → Pool → Dense(64) → Dense(10) | 167,562 | **69.28%** | 0.9122 |

> **CNN outperforms ANN by +20 percentage points using 19× fewer parameters.**

## Key Findings
- The ANN struggles most on visually similar classes: deer (F1=0.32), dog (0.32), cat (0.38)
- The CNN shows mild overfitting (~7% train/val gap) but dramatically better generalisation
- Convolutional filters learn spatial hierarchies that Dense layers fundamentally cannot capture

## Project Structure
```
06-ann-cnn-cifar10/
├── cifar10_ann_vs_cnn.ipynb   # Main notebook
├── requirements.txt
├── .gitignore
└── README.md
```

## How to Run
```bash
# 1. Clone the repo
git clone https://github.com/SaviortheCoder/machine-learning-portfolio.git
cd machine-learning-portfolio/06-ann-cnn-cifar10

# 2. Install dependencies
pip install -r requirements.txt

# 3. Open the notebook
jupyter notebook cifar10_ann_vs_cnn.ipynb
```

## Tech Stack
- Python 3.11 · TensorFlow/Keras 2.21 · NumPy · Matplotlib · scikit-learn · visualkeras

## Dataset
[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) — 60,000 32×32 colour images across 10 classes (50k train / 10k test), loaded directly via `tensorflow.keras.datasets`.
