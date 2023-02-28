"""K means clustering model"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


# Name data source
DATA_SOURCE = 'Mall_Customers.csv'

# Load dataset from csv
dataset = pd.read_csv(DATA_SOURCE)
X = dataset.iloc[:, [3,4]].values

# # Encode categorical data
# ct = ColumnTransformer(
#     transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
# X = np.array(ct.fit_transform(X))

# Elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++",
                    random_state=42, n_init="auto")
    kmeans.fit(X)
    inertia = kmeans.inertia_
    wcss.append(int(inertia))

# plt.plot(range(1,11), wcss)
# plt.title("Elbow method")
# plt.xlabel("Number of clusters")
# plt.ylabel("WCSS")
# plt.show()

# Visualise kclusters
kmeans = KMeans(n_clusters=5, init="k-means++",
                random_state=42, n_init="auto")

y_means = kmeans.fit_predict(X)

p_size=25
c_size=100
plt.scatter(X[y_means == 0, 0], X[y_means == 0, 1],
            s=p_size, c="red", label="Cluster1")
plt.scatter(X[y_means == 1, 0], X[y_means == 1, 1],
            s=p_size, c="blue", label="Cluster2")
plt.scatter(X[y_means == 2, 0], X[y_means == 2, 1],
            s=p_size, c="green", label="Cluster3")
plt.scatter(X[y_means == 3, 0], X[y_means == 3, 1],
            s=p_size, c="cyan", label="Cluster4")
plt.scatter(X[y_means == 4, 0], X[y_means == 4, 1],
            s=p_size, c="magenta", label="Cluster5")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=c_size, c="black", label="Centroids")
plt.title("Clusters")
plt.xlabel("Income")
plt.ylabel("Spend score")
plt.legend()
plt.show()