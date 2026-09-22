# ==========================================
# ASSIGNMENT 2
# Apply PCA to reduce dataset dimensions
# ==========================================

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# Load Iris dataset
iris = load_iris()
X = iris.data

# Reduce 4 dimensions to 2 dimensions
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Print reduced dataset
print("Original Shape:", X.shape)
print("Reduced Shape:", X_pca.shape)

print("\nPCA Data:")
print(X_pca)

# Print explained variance ratio
print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)