"""Hierarchical clustering model"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering


# Name data source
DATA_SOURCE = 'Mall_Customers.csv'

# Load dataset from csv
dataset = pd.read_csv(DATA_SOURCE)
X = dataset.iloc[:, [3, 4]].values

#Using Dendrogram to find optimal number of clusters
dendrogram = sch.dendrogram(sch.linkage(X, method="ward"))
plt.title("Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Euclidiean distance")
plt.show()


# Visualise kclusters
hc = AgglomerativeClustering(n_clusters=5, affinity="euclidean", linkage="ward")
y_hc = hc.fit_predict(X)

p_size = 25
c_size = 100
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1],
            s=p_size, c="red", label="Cluster1")
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1],
            s=p_size, c="blue", label="Cluster2")
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1],
            s=p_size, c="green", label="Cluster3")
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1],
            s=p_size, c="cyan", label="Cluster4")
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1],
            s=p_size, c="magenta", label="Cluster5")


plt.title("Clusters")
plt.xlabel("Income")
plt.ylabel("Spend score")
plt.legend()
plt.show()
