# ==========================================
# MINI PROJECT 3
# Iris Flower Clustering
# ==========================================

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load Iris dataset
iris = load_iris()

X = iris.data
y = iris.target

# Apply K-Means
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)

# Predicted clusters
print("Predicted Clusters:")
print(clusters)

# Cluster centers
print("\nCluster Centers:")
print(kmeans.cluster_centers_)

# Visualize predicted clusters
plt.scatter(X[:, 0], X[:, 1], c=clusters)

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Iris Flower - K-Means Clustering")

plt.show()

# Compare true labels and predicted clusters
print("\nTrue Labels:")
print(y)

print("\nPredicted Clusters:")
print(clusters)