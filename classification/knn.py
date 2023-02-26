"""Logistic regression model"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

# Name data source
DATA_SOURCE = 'Social_Network_Ads.csv'

# Load dataset from csv
dataset = pd.read_csv(DATA_SOURCE)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

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

X_set, y_set = sc.inverse_transform(X_test), y_test
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 10, stop=X_set[:, 0].max() + 10, step=0.25),
                     np.arange(start=X_set[:, 1].min() - 1000, stop=X_set[:, 1].max() + 1000, step=0.25))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
plt.title('Logistic Regression')
plt.xlabel('Age')
plt.ylabel("Salary")
plt.legend()
plt.show()
