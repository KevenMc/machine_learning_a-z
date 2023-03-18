"""Support Vector Machine classifier"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score, GridSearchCV


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

# Train SVR model on data
classifier = SVC(kernel="rbf", random_state=0)
classifier.fit(X_train, y_train)

# Predict test set
y_pred = classifier.predict(X_test)

#Apply k-fold cross validation
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
print(f"Accuracy: {accuracies.mean()*100:.2f} %")
print(f"Standard deviation: {accuracies.std()*100:.2f}")


#Grid Search
parameters=[{'C':[0.25,0.5,0.75,1],
             'kernel': ['linear']},
            {'C':[0.25,0.5,0.75,1],
             'kernel': ['rbf'],
             'gamma': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}]

grid_search = GridSearchCV(estimator=classifier,
                           param_grid=parameters,
                           scoring='accuracy',
                           cv=10,
                           n_jobs=-1)

grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_params = grid_search.best_params_

print(f"Accuracy: {best_accuracy*100:.2f} %")
print(f"Best params: {best_params}")
