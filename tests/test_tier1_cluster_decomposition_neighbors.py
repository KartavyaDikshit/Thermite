import numpy as np
from tests.conftest import get_module

# Dynamically import required modules
cluster = get_module("cluster")
decomposition = get_module("decomposition")
neighbors = get_module("neighbors")


# KMeans tests
def test_kmeans_fit_predict():
    """Basic KMeans clustering, check cluster centers and labels."""
    km = cluster.KMeans(n_clusters=2, random_state=42, n_init=3)
    X = np.array([[1.0, 2.0], [1.5, 1.8], [10.0, 12.0], [11.0, 11.5]])
    
    labels = km.fit_predict(X)
    assert labels.shape == (4,)
    assert set(labels) == {0, 1}
    assert km.cluster_centers_.shape == (2, 2)


def test_kmeans_init():
    """Check KMeans with different init values."""
    X = np.array([[1.0, 2.0], [1.5, 1.8], [10.0, 12.0], [11.0, 11.5]])
    for init in ["k-means++", "random"]:
        km = cluster.KMeans(n_clusters=2, init=init, random_state=42, n_init=3)
        km.fit(X)
        assert km.labels_.shape == (4,)


def test_kmeans_n_clusters():
    """Verify n_clusters config matches fitted cluster centers."""
    X = np.array([[1.0, 2.0], [1.5, 1.8], [10.0, 12.0], [11.0, 11.5], [5.0, 5.0]])
    km = cluster.KMeans(n_clusters=3, random_state=42, n_init=3)
    km.fit(X)
    assert km.cluster_centers_.shape == (3, 2)


def test_kmeans_inertia():
    """Check inertia is a float and is non-negative."""
    km = cluster.KMeans(n_clusters=2, random_state=42, n_init=3)
    X = np.array([[1.0, 2.0], [1.5, 1.8], [10.0, 12.0], [11.0, 11.5]])
    km.fit(X)
    
    assert isinstance(km.inertia_, float)
    assert km.inertia_ >= 0.0


def test_kmeans_transform():
    """Verify transform method returns distance matrix of shape (n_samples, n_clusters)."""
    km = cluster.KMeans(n_clusters=2, random_state=42, n_init=3)
    X = np.array([[1.0, 2.0], [1.5, 1.8], [10.0, 12.0]])
    km.fit(X)
    
    dists = km.transform(X)
    assert dists.shape == (3, 2)


# DBSCAN tests
def test_dbscan_fit():
    """Basic DBSCAN clustering."""
    db = cluster.DBSCAN(eps=1.5, min_samples=2)
    X = np.array([[1.0, 2.0], [1.1, 2.1], [10.0, 12.0], [10.1, 11.9]])
    
    db.fit(X)
    assert db.labels_.shape == (4,)
    # There should be two clear clusters, plus no noise points
    assert len(set(db.labels_) - {-1}) == 2


def test_dbscan_core_samples():
    """Verify DBSCAN core sample attributes."""
    db = cluster.DBSCAN(eps=1.5, min_samples=2)
    X = np.array([[1.0, 2.0], [1.1, 2.1], [10.0, 12.0]])
    db.fit(X)
    
    assert db.core_sample_indices_.shape[0] <= X.shape[0]
    assert db.components_.shape[1] == X.shape[1]


def test_dbscan_noise():
    """Verify DBSCAN assigns label -1 to noise/outlier points."""
    db = cluster.DBSCAN(eps=1.0, min_samples=2)
    # The last point is far away from the others
    X = np.array([[1.0, 2.0], [1.1, 2.1], [10.0, 12.0]])
    db.fit(X)
    
    assert db.labels_[2] == -1


def test_dbscan_metric():
    """Check DBSCAN with different metrics."""
    X = np.array([[1.0, 2.0], [1.1, 2.1], [10.0, 12.0], [10.1, 11.9]])
    for metric in ["euclidean", "manhattan"]:
        db = cluster.DBSCAN(eps=2.0, min_samples=2, metric=metric)
        db.fit(X)
        assert db.labels_.shape == (4,)


def test_dbscan_eps_effect():
    """Smaller eps should result in more noise points."""
    X = np.array([[1.0, 2.0], [1.5, 2.5], [2.0, 3.0]])
    
    db_large = cluster.DBSCAN(eps=2.0, min_samples=2)
    db_small = cluster.DBSCAN(eps=0.1, min_samples=2)
    
    db_large.fit(X)
    db_small.fit(X)
    
    # Large eps should group them, small eps should make all of them noise (-1)
    assert np.sum(db_large.labels_ == -1) < np.sum(db_small.labels_ == -1)


# PCA tests
def test_pca_fit_transform():
    """Basic PCA dimensionality reduction."""
    pca = decomposition.PCA(n_components=2)
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
    
    X_trans = pca.fit_transform(X)
    assert X_trans.shape == (4, 2)


def test_pca_explained_variance():
    """Check explained_variance_ratio_."""
    pca = decomposition.PCA(n_components=2)
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
    pca.fit(X)
    
    ratios = pca.explained_variance_ratio_
    assert ratios.shape == (2,)
    assert np.sum(ratios) <= 1.0001
    assert all(ratios >= 0.0)


def test_pca_components():
    """Check shape of components_ is (n_components, n_features)."""
    pca = decomposition.PCA(n_components=1)
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    pca.fit(X)
    
    assert pca.components_.shape == (1, 2)


def test_pca_inverse_transform():
    """Check reconstruction of original space."""
    pca = decomposition.PCA(n_components=2)
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    X_trans = pca.fit_transform(X)
    X_inv = pca.inverse_transform(X_trans)
    
    # We should have perfect or near-perfect reconstruction since n_components=2 is enough for 3 collinear-ish points
    np.testing.assert_allclose(X, X_inv, atol=1e-5)


def test_pca_singular_values():
    """Check singular_values_ are non-negative and matching shape."""
    pca = decomposition.PCA(n_components=2)
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]])
    pca.fit(X)
    
    assert pca.singular_values_.shape == (2,)
    assert all(pca.singular_values_ >= 0.0)


# KNeighborsClassifier tests
def test_knn_classifier_fit_predict():
    """Basic KNN classification."""
    knn = neighbors.KNeighborsClassifier(n_neighbors=3)
    X = np.array([[1.0, 1.0], [1.1, 1.1], [1.2, 1.2], [5.0, 5.0], [5.1, 5.1], [5.2, 5.2]])
    y = np.array([0, 0, 0, 1, 1, 1])
    
    knn.fit(X, y)
    pred = knn.predict([[1.05, 1.05], [4.9, 4.9]])
    np.testing.assert_array_equal(pred, [0, 1])


def test_knn_classifier_predict_proba():
    """Check KNN predict_proba."""
    knn = neighbors.KNeighborsClassifier(n_neighbors=2)
    X = np.array([[1.0, 1.0], [1.1, 1.1], [5.0, 5.0], [5.1, 5.1]])
    y = np.array([0, 1, 0, 1])
    knn.fit(X, y)
    
    # Class 0 and 1 are mixed, test sample close to X[0] and X[1]
    proba = knn.predict_proba([[1.05, 1.05]])
    assert proba.shape == (1, 2)
    # One neighbor is class 0, one neighbor is class 1
    np.testing.assert_allclose(proba[0], [0.5, 0.5])


def test_knn_classifier_k_neighbors():
    """Verify kneighbors method outputs shapes."""
    knn = neighbors.KNeighborsClassifier(n_neighbors=2)
    X = np.array([[1.0, 1.0], [1.1, 1.1], [5.0, 5.0]])
    y = np.array([0, 0, 1])
    knn.fit(X, y)
    
    dists, indices = knn.kneighbors([[1.0, 1.0]])
    assert dists.shape == (1, 2)
    assert indices.shape == (1, 2)
    assert indices[0, 0] == 0  # Nearest neighbor should be itself (index 0)


def test_knn_classifier_weights():
    """Check KNN weights parameter combinations."""
    X = np.array([[1.0, 1.0], [1.1, 1.1], [5.0, 5.0]])
    y = np.array([0, 0, 1])
    
    for weight in ["uniform", "distance"]:
        knn = neighbors.KNeighborsClassifier(n_neighbors=2, weights=weight)
        knn.fit(X, y)
        assert knn.score(X, y) > 0.5


def test_knn_classifier_algorithm():
    """Check KNN with different algorithm configurations."""
    X = np.array([[1.0, 1.0], [1.1, 1.1], [5.0, 5.0]])
    y = np.array([0, 0, 1])
    
    for algo in ["brute", "kd_tree", "ball_tree"]:
        knn = neighbors.KNeighborsClassifier(n_neighbors=2, algorithm=algo)
        knn.fit(X, y)
        pred = knn.predict(X)
        assert pred.shape == (3,)
