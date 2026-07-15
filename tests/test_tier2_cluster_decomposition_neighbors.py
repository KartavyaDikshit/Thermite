import numpy as np
import pytest
from tests.conftest import get_module

# Dynamically import cluster, decomposition, and neighbors modules
cluster = get_module("cluster")
decomposition = get_module("decomposition")
neighbors = get_module("neighbors")


# =====================================================================
# KMeans Boundary Cases
# =====================================================================

def test_kmeans_empty_input():
    """1. Fitting on empty array raises ValueError."""
    km = cluster.KMeans()
    with pytest.raises(ValueError):
        km.fit(np.empty((0, 2)))


def test_kmeans_more_clusters_than_samples():
    """2. Fitting with n_clusters > n_samples raises ValueError."""
    km = cluster.KMeans(n_clusters=5)
    with pytest.raises(ValueError):
        km.fit([[1.0], [2.0]])


def test_kmeans_single_sample():
    """3. Fitting 1 sample with n_clusters=1 succeeds and center matches sample."""
    km = cluster.KMeans(n_clusters=1, n_init=1)
    X = np.array([[10.0, 20.0]])
    km.fit(X)
    
    np.testing.assert_array_almost_equal(km.cluster_centers_, X)
    assert km.labels_[0] == 0


def test_kmeans_zero_variance_features():
    """4. Zero variance features (constant columns) should fit successfully."""
    km = cluster.KMeans(n_clusters=2, n_init=1, random_state=42)
    X = np.array([[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]])
    # Since all samples are identical, they should all be in cluster 0
    km.fit(X)
    np.testing.assert_array_equal(km.labels_, [0, 0, 0])


def test_kmeans_invalid_params():
    """5. Invalid parameter n_clusters=0 should raise ValueError."""
    with pytest.raises(ValueError):
        km = cluster.KMeans(n_clusters=0)
        km.fit([[1.0], [2.0]])


# =====================================================================
# DBSCAN Boundary Cases
# =====================================================================

def test_dbscan_empty_input():
    """1. Fitting on empty array raises ValueError."""
    db = cluster.DBSCAN()
    with pytest.raises(ValueError):
        db.fit(np.empty((0, 2)))


def test_dbscan_single_sample():
    """2. Fitting single sample with min_samples=2 should mark it as noise (-1)."""
    db = cluster.DBSCAN(min_samples=2)
    X = np.array([[1.0, 2.0]])
    db.fit(X)
    
    assert list(db.labels_) == [-1]


def test_dbscan_all_noise():
    """3. DBSCAN where all samples are noise due to small eps."""
    db = cluster.DBSCAN(eps=0.01, min_samples=2)
    X = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
    db.fit(X)
    
    np.testing.assert_array_equal(db.labels_, [-1, -1, -1])


def test_dbscan_zero_variance_features():
    """4. Zero variance features (constant columns) fits all in one cluster if min_samples met."""
    db = cluster.DBSCAN(eps=1.0, min_samples=2)
    X = np.array([[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]])
    db.fit(X)
    
    np.testing.assert_array_equal(db.labels_, [0, 0, 0])


def test_dbscan_extreme_eps():
    """5. Extremely large eps should group all samples into a single cluster (if min_samples met)."""
    db = cluster.DBSCAN(eps=1e10, min_samples=2)
    X = np.array([[1.0, 2.0], [100.0, 200.0]])
    db.fit(X)
    
    np.testing.assert_array_equal(db.labels_, [0, 0])


# =====================================================================
# PCA Boundary Cases
# =====================================================================

def test_pca_empty_input():
    """1. Fitting on empty array raises ValueError."""
    pca = decomposition.PCA()
    with pytest.raises(ValueError):
        pca.fit(np.empty((0, 2)))


def test_pca_too_many_components():
    """2. Fitting with n_components > min(samples, features) raises ValueError."""
    pca = decomposition.PCA(n_components=5)
    with pytest.raises(ValueError):
        pca.fit([[1.0, 2.0], [3.0, 4.0]])


def test_pca_single_sample():
    """3. PCA on a single sample succeeds with warning and explained_variance_ contains NaN."""
    pca = decomposition.PCA(n_components=1)
    # Wrap in warnings block to suppress the division-by-zero RuntimeWarning
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pca.fit([[1.0, 2.0]])
    
    assert len(pca.explained_variance_) == 1
    assert np.isnan(pca.explained_variance_[0])


def test_pca_zero_variance_features():
    """4. Fitting PCA on constant features should result in 0 explained variance."""
    pca = decomposition.PCA(n_components=1)
    X = np.array([[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]])
    
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pca.fit(X)
        
    np.testing.assert_array_almost_equal(pca.explained_variance_, [0.0])


def test_pca_invalid_components():
    """5. Negative n_components raises ValueError."""
    with pytest.raises(ValueError):
        pca = decomposition.PCA(n_components=-1)
        pca.fit([[1.0, 2.0], [3.0, 4.0]])


# =====================================================================
# KNeighborsClassifier Boundary Cases
# =====================================================================

def test_knn_empty_input():
    """1. Fitting on empty array raises ValueError."""
    knn = neighbors.KNeighborsClassifier()
    with pytest.raises(ValueError):
        knn.fit(np.empty((0, 2)), np.empty(0))


def test_knn_fewer_samples_than_neighbors():
    """2. Predicting with n_neighbors > n_samples_fit raises ValueError."""
    knn = neighbors.KNeighborsClassifier(n_neighbors=5)
    knn.fit([[1.0], [2.0]], [0, 1])
    with pytest.raises(ValueError):
        knn.predict([[1.5]])


def test_knn_single_class_target():
    """3. Fitting and predicting with a single class target works successfully."""
    knn = neighbors.KNeighborsClassifier(n_neighbors=1)
    knn.fit([[1.0], [2.0]], [1, 1])
    
    pred = knn.predict([[1.5]])
    np.testing.assert_array_equal(pred, [1])


def test_knn_invalid_neighbors():
    """4. Zero or negative n_neighbors raises ValueError."""
    with pytest.raises(ValueError):
        knn = neighbors.KNeighborsClassifier(n_neighbors=0)
        knn.fit([[1.0], [2.0]], [0, 1])


def test_knn_extreme_distances():
    """5. Predicting on points extremely far from training data should still work and select nearest neighbors."""
    knn = neighbors.KNeighborsClassifier(n_neighbors=1)
    knn.fit([[0.0], [1.0]], [0, 1])
    
    pred = knn.predict([[1e10]])
    np.testing.assert_array_equal(pred, [1])
