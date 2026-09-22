# ==========================================
# ASSIGNMENT 1
# Perform K-Means on Iris dataset
# and visualize clusters
# ==========================================

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load Iris dataset
iris = load_iris()
X = iris.data

# Apply K-Means with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)

# Print cluster centers
print("Cluster Centers:")
print(kmeans.cluster_centers_)

# Visualize clusters
plt.scatter(X[:, 0], X[:, 1], c=clusters)

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("K-Means Clusters on Iris Dataset")

plt.show()