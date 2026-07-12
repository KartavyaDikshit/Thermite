import numpy as np
import scipy.sparse as sp
from thermite.cluster import KMeans

# Dense data
np.random.seed(42)
X_dense = np.random.rand(100, 5)
kmeans_dense = KMeans(n_clusters=3, random_state=42)
kmeans_dense.fit(X_dense)
preds_dense = kmeans_dense.predict(X_dense)
print("Dense centers shape:", kmeans_dense.cluster_centers_.shape)

# Sparse data
X_sparse = sp.csr_matrix(X_dense)
kmeans_sparse = KMeans(n_clusters=3, random_state=42)
kmeans_sparse.fit(X_sparse)
preds_sparse = kmeans_sparse.predict(X_sparse)

print("Sparse centers shape:", kmeans_sparse.cluster_centers_.shape)
print("Centers match:", np.allclose(kmeans_dense.cluster_centers_, kmeans_sparse.cluster_centers_))
print("Predictions match:", np.array_equal(preds_dense, preds_sparse))
