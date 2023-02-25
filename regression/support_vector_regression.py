"""Support Vector Machine model"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Name data source
data_source = 'Position_Salaries.csv'

# Load dataset from csv
dataset = pd.read_csv(data_source)
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
y = y.reshape(len(y), 1)

# Feature scaling
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# Train SVR model on data
regressor = SVR(kernel="rbf")
regressor.fit(X, y)

# Predict
y_pred = sc_y.inverse_transform(regressor.predict(
    sc_X.transform([[6.5]])).reshape(-1, 1))
print(y_pred)

#Inverse scale
X_ = sc_X.inverse_transform(X)
y_ = sc_y.inverse_transform(y)

# Visualise Polynomial Linear regression
plt.scatter(X_, y_, color="red")
plt.plot(X_, sc_y.inverse_transform(
    regressor.predict(X).reshape(-1, 1)), color="blue")
plt.title("Truth or Bluff (SVR)")
plt.xlabel("Poisition leven")
plt.ylabel("salary")
plt.show()

# Visualise smoother line
X_grid = np.arange(min(X_), max(X_), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_, y_, color="red")
plt.plot(X_grid, sc_y.inverse_transform(
    regressor.predict(sc_X.transform(X_grid)).reshape(-1, 1)), color="blue")
plt.title("Truth or Bluff (SVR)")
plt.xlabel("Poisition leven")
plt.ylabel("salary")
plt.show()
