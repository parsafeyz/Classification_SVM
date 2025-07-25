# Support Vector Machine (SVM) Classifier 🧠✨

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/parsafeyz/Classification_SVM/blob/main/SVM_v1.ipynb)

This notebook explores how to use **Support Vector Machines (SVM)** with the RBF kernel to build a classifier using scikit-learn. Perfect for those looking to level up their ML game with a clean and classic algorithm. 💻📊

## 🚀 Features

- Loads and preprocesses a dataset using `pandas` and `sklearn.preprocessing`
- Splits the data into training and test sets using `train_test_split`
- Trains an SVM classifier with the **RBF kernel**
- Evaluates performance using:
  - Confusion Matrix (with a cute visualization 🎨)
  - F1 Score
  - Jaccard Index

## 🧩 Libraries Used

```python
import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
import itertools
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_score
%matplotlib inline
