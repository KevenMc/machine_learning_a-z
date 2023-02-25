"""Polynomial regression model"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# Name data source
DATA_SOURCE = 'Position_Salaries.csv'

# Load dataset from csv
dataset = pd.read_csv(DATA_SOURCE)
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values


# Train simple linear regression model
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Train polynomial regression model
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)

# Train simple linear regression model for polynomial
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)


# Visualise Linear regression
# plt.scatter(X, y, color="red")
# plt.plot(X, lin_reg.predict(X), color="blue")
# plt.title("Truth or Bluff (lin_reg)")
# plt.xlabel("Poisition leven")
# plt.ylabel("salary")
# plt.show()


# Visualise Polynomial Linear regression
# plt.scatter(X, y, color="red")
# plt.plot(X, lin_reg2.predict(X_poly), color="blue")
# plt.title("Truth or Bluff (lin_reg2)")
# plt.xlabel("Poisition leven")
# plt.ylabel("salary")
# plt.show()

#Visualise smoother line
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid),1) )
plt.scatter(X, y, color="red")
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), color="blue")
plt.title("Truth or Bluff (lin_reg2)")
plt.xlabel("Poisition leven")
plt.ylabel("salary")
plt.show()

print(f"Lin prediction: {lin_reg.predict([[6.5]])[0]}")
print(f"Poly prediction: {lin_reg2.predict(poly_reg.fit_transform([[6.5]]))[0]}")
