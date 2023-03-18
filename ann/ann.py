"""Artificial neural network"""

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

import tensorflow as tf

# Name data source
DATA_SOURCE = 'Churn_Modelling.csv'

# Load dataset from csv
dataset = pd.read_csv(DATA_SOURCE)
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

#Label encode country and gender
le = LabelEncoder()
X[:,2] = le.fit_transform(X[:,2])

# Encode categorical data
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))


# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(6, activation='relu', input_shape=(12,)),
    tf.keras.layers.Dense(6, activation='relu'),

    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with your input data
model.fit(X_train, y_train, epochs=20, batch_size=32)


predicted_probabilities = model.predict(X_test)

threshold = 0.5
y_pred = np.where(predicted_probabilities > threshold, 1, 0)

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
