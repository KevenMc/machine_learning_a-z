"""K nearest neighbours model"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

# Name data source
DATA_SOURCE = 'Data.csv'

# Load dataset from csv
dataset = pd.read_csv(DATA_SOURCE)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values /2 -1

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train simple logistic regression model
classifier = KNeighborsClassifier()
classifier.fit(X_train, y_train)

# Predict test set
y_pred = classifier.predict(X_test)

# Confusion matrix and fscore
m = confusion_matrix(y_pred.reshape(len(y_pred), 1),
                     y_test.reshape(len(y_test), 1))
f = f1_score(y_pred.reshape(len(y_pred), 1),
             y_test.reshape(len(y_test), 1))
a = accuracy_score(y_pred.reshape(len(y_pred), 1),
                   y_test.reshape(len(y_test), 1))

print(f"Accuracy (fscore): {f}")
print(f"Accuracy (accuracy score): {a}")
print(f"Confusion matrix:\n {m}")
