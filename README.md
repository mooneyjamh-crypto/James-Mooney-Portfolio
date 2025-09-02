# STAT 426 Final Project – Fashion-MNIST Classification

This repository contains my final project for **STAT 426: Introduction to Data Science and Machine Learning**.  
The project investigates and compares the performance of seven supervised machine learning models on the **Fashion-MNIST dataset** (70,000 28×28 grayscale images across 10 fashion categories).

## Contents
- `STAT426final.ipynb` – Jupyter notebook containing all code, data preparation, model training, and evaluations.
- `STAT426 Final Report.pdf` – Formal written report summarizing methodology, results, and analysis.

## Models Implemented
- **K-Nearest Neighbors (KNN)**
- **Support Vector Machine (Linear)**
- **Support Vector Machine (Nonlinear, RBF kernel)**
- **Random Forest**
- **Linear Discriminant Analysis (LDA)**
- **Quadratic Discriminant Analysis (QDA)**
- **Convolutional Neural Network (CNN)**

## Results
- PCA reduced dimensionality by **91%** (from 784 to 69 features) while preserving 90% of variance.
- Among non-deep-learning models, the **Nonlinear SVM (RBF)** achieved the best performance with **80.74% accuracy**.
- The **Convolutional Neural Network (CNN)** significantly outperformed all other methods, reaching **90.63% accuracy** when trained on the full dataset.

## Tools & Libraries
- Python (NumPy, pandas, scikit-learn, PyTorch, torchvision, matplotlib)
- Jupyter Notebook

## About
In this project I implemented and evaluated seven supervised learning models on a 70,000 image dataset. I leveraged PCA to compress data by over 91% while retaining 90% of variance. Ultimately, I found the Convolutional Neural Network, achieving test accuracy of 90.63%, to be the most effective method, outperforming traditional machine learning methods by up to 9-15%.
