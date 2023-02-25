"""Simple linear regression model"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#Name data source
DATA_SOURCE = 'Salary_Data.csv'

# Load dataset from csv
dataset = pd.read_csv(DATA_SOURCE)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)

#Train simple linear regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predict test set
y_pred = regressor.predict(X_test)
print(y_pred)

#Visualise training data
plt.scatter(X_train, y_train, color="red")
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.title("Salary vs experience (training set)")
plt.xlabel("Years of experience")
plt.ylabel("salary")
plt.show()

# Visualise test data
plt.scatter(X_test, y_test, color="red")
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.title("Salary vs experience (test set)")
plt.xlabel("Years of experience")
plt.ylabel("salary")
plt.show()
