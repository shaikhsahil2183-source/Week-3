from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

iris = load_iris()
X = iris.data

pca = PCA(n_components=2)
pca.fit(X)

print("Explained Variance Ratio:")
print(pca.explained_variance_ratio_)