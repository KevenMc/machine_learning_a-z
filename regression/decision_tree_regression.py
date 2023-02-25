"""Decision Tree regression"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# Name data source
data_source = 'Position_Salaries.csv'

# Load dataset from csv
dataset = pd.read_csv(data_source)
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Train decision tree
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)

# Predict
y_pred = regressor.predict([[6.5]])
print(y_pred)

# Visualise smoother line
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color="red")
plt.plot(X_grid, regressor.predict(X_grid), color="blue")
plt.title("Truth or Bluff (decision_tree)")
plt.xlabel("Poisition leven")
plt.ylabel("salary")
plt.show()
