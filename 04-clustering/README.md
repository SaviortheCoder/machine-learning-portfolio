# 04 · Customer Segmentation — Clustering

## Problem
Unsupervised segmentation of HR candidates into behavioral personas without using labels.

## Dataset
**HR Analytics Job Change** (hrdata.csv) — used without the target label.

## What This Notebook Does
1. Preprocessing — encoding and scaling (required for distance-based methods)
2. KMeans + Elbow Method to find optimal k
3. Agglomerative Clustering (Ward linkage) + dendrogram
4. PCA projection — 2D cluster visualization
5. Cluster profiling and business persona descriptions

## Key Results
- Elbow method: k=4 optimal
- Cluster 2: experienced, high-development-city candidates → highest job-change probability
- Agglomerative clustering produces more balanced segment sizes than KMeans

## How to Run
```bash
cd 04-clustering
jupyter notebook notebook.ipynb
```
