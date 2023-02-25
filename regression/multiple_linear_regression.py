"""Multiple linear regression model"""

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Name data source
data_source = '50_Startups.csv'

# Load dataset from csv
dataset = pd.read_csv(data_source)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Encode categorical data: state name
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)

# Train simple linear regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict test set
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)

print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))
print((y_pred/y_test).reshape(len(y_pred), 1))
