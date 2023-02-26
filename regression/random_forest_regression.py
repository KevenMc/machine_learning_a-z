"""Random Forest regression"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# Name data source
DATA_SOURCE = 'Position_Salaries.csv'

# Load dataset from csv
dataset = pd.read_csv(DATA_SOURCE)
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Train decision tree
regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(X, y)

# Predict
y_pred = regressor.predict([[6.5]])
print(y_pred)

# Visualise smoother line
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color="red")
plt.plot(X_grid, regressor.predict(X_grid), color="blue")
plt.title("Truth or Bluff (random_forest)")
plt.xlabel("Poisition level")
plt.ylabel("salary")
plt.show()
