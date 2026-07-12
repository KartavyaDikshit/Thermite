import numpy as np
from . import _core

class KMeans:
    def __init__(self, n_clusters=8, *, max_iter=300, tol=1e-4, n_init=10, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state
        self._model = _core.KMeans(
            n_clusters=n_clusters,
            max_iter=max_iter,
            tol=tol,
            n_init=n_init,
            random_state=random_state
        )

    def fit(self, X, y=None):
        import scipy.sparse as sp
        if sp.issparse(X):
            X_csr = X.tocsr()
            data = np.asarray(X_csr.data, dtype=np.float64)
            indices = np.asarray(X_csr.indices, dtype=np.uintp)
            indptr = np.asarray(X_csr.indptr, dtype=np.uintp)
            rows, cols = X_csr.shape
            self._model.fit_sparse(data, indices, indptr, rows, cols)
            return self
            
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        self._model.fit(X)
        return self

    def predict(self, X):
        import scipy.sparse as sp
        if sp.issparse(X):
            X_csr = X.tocsr()
            data = np.asarray(X_csr.data, dtype=np.float64)
            indices = np.asarray(X_csr.indices, dtype=np.uintp)
            indptr = np.asarray(X_csr.indptr, dtype=np.uintp)
            rows, cols = X_csr.shape
            return self._model.predict_sparse(data, indices, indptr, rows, cols)

        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        return self._model.predict(X)

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.predict(X)

    @property
    def cluster_centers_(self):
        return self._model.cluster_centers_

    @property
    def labels_(self):
        return self._model.labels_

    @property
    def inertia_(self):
        return self._model.inertia_

    @property
    def n_iter_(self):
        return self._model.n_iter_


class DBSCAN:
    def __init__(self, eps=0.5, *, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self._model = _core.DBSCAN(
            eps=eps,
            min_samples=min_samples
        )

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        self._model.fit(X)
        return self

    def fit_predict(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        return self._model.fit_predict(X)

    @property
    def labels_(self):
        return self._model.labels_

    @property
    def core_sample_indices_(self):
        return self._model.core_sample_indices_
