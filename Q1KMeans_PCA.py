from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

iris = load_iris()
X = iris.data

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X)

print("Cluster Centers:")
print(kmeans.cluster_centers_)