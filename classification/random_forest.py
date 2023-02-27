"""Random forest of decision trees"""

# Name data source
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

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

# Train Random Forest model on data
classifier = RandomForestClassifier(
    n_estimators=5,
    criterion="entropy",
    random_state=0)
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
