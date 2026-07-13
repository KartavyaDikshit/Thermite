import numpy as np
import warnings
from . import _core

def _catch_panic(func):
    def wrapper(self, *args, **kwargs):
        # basic input validation for all estimators
        for arg in args:
            if isinstance(arg, np.ndarray) and arg.size == 0:
                raise ValueError("Empty input")
            if isinstance(arg, (list, tuple)) and len(arg) == 0:
                raise ValueError("Empty input")
        
        try:
            return func(self, *args, **kwargs)
        except BaseException as e:
            err_str = str(e).lower()
            if "panic" in err_str or "empty" in err_str or "bounds" in err_str or "singular" in err_str or "invalid" in err_str:
                raise ValueError(str(e))
            raise
    return wrapper


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

    @_catch_panic
    def fit(self, X, y=None):
        import scipy.sparse as sp
        if sp.issparse(X):
            X_csr = X.tocsr()
            data = np.ascontiguousarray(np.asarray(X_csr.data, dtype=np.float64))
            indices = np.ascontiguousarray(np.asarray(X_csr.indices, dtype=np.uintp))
            indptr = np.ascontiguousarray(np.asarray(X_csr.indptr, dtype=np.uintp))
            rows, cols = X_csr.shape
            self._model.fit_sparse(data, indices, indptr, rows, cols)
            return self
            
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        self._model.fit(X)
        return self

    @_catch_panic
    def predict(self, X):
        import scipy.sparse as sp
        if sp.issparse(X):
            X_csr = X.tocsr()
            data = np.ascontiguousarray(np.asarray(X_csr.data, dtype=np.float64))
            indices = np.ascontiguousarray(np.asarray(X_csr.indices, dtype=np.uintp))
            indptr = np.ascontiguousarray(np.asarray(X_csr.indptr, dtype=np.uintp))
            rows, cols = X_csr.shape
            return self._model.predict_sparse(data, indices, indptr, rows, cols)

        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        return self._model.predict(X)

    @_catch_panic
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

    @_catch_panic
    def fit(self, X, y=None):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        self._model.fit(X)
        return self

    @_catch_panic
    def fit_predict(self, X, y=None):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        return self._model.fit_predict(X)

    @property
    def labels_(self):
        return self._model.labels_

    @property
    def core_sample_indices_(self):
        return self._model.core_sample_indices_

class MiniBatchKMeans:
    def __init__(self, n_clusters=8, *, max_iter=100, batch_size=1024, tol=0.0):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.tol = tol
        self._model = _core.MiniBatchKMeans(
            n_clusters=n_clusters, max_iter=max_iter, batch_size=batch_size, tol=tol
        )

    @_catch_panic
    def fit(self, X, y=None):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        self._model.fit(X)
        return self

    @_catch_panic
    def partial_fit(self, X, y=None):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        self._model.partial_fit(X)
        return self

    @_catch_panic
    def predict(self, X):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        return self._model.predict(X)

    @_catch_panic
    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.predict(X)

    @property
    def cluster_centers_(self):
        return self._model.cluster_centers_

class SpectralClustering:
    def __init__(self, n_clusters=8, *, eigen_solver=None, n_components=None, random_state=None, n_init=10, gamma=1.0, affinity='rbf', n_neighbors=10, eigen_tol='auto', assign_labels='kmeans', degree=3, coef0=1, kernel_params=None, n_jobs=None):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self._model = _core.SpectralClustering(
            n_clusters=n_clusters, random_state=random_state
        )

    @_catch_panic
    def fit_predict(self, X, y=None):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        return self._model.fit_predict(X)

class AffinityPropagation:
    def __init__(self, *, damping=0.5, max_iter=200, convergence_iter=15, copy=True, preference=None, affinity='euclidean', verbose=False, random_state=None):
        self.damping = damping
        self.max_iter = max_iter
        self.convergence_iter = convergence_iter
        self.copy = copy
        self.preference = preference
        self.affinity = affinity
        self.verbose = verbose
        self.random_state = random_state
        self._model = _core.AffinityPropagation(
            damping=damping,
            max_iter=max_iter,
            convergence_iter=convergence_iter,
            copy=copy,
            preference=preference,
            affinity=affinity,
            verbose=verbose,
            random_state=random_state
        )

    @_catch_panic
    def fit(self, X, y=None):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        self._model.fit(X)
        return self

    @_catch_panic
    def predict(self, X):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        return self._model.predict(X)

    @_catch_panic
    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.predict(X)

class MeanShift:
    def __init__(self, *, bandwidth=None, seeds=None, bin_seeding=False, min_bin_freq=1, cluster_all=True, n_jobs=None, max_iter=300):
        self.bandwidth = bandwidth
        self.seeds = seeds
        self.bin_seeding = bin_seeding
        self.min_bin_freq = min_bin_freq
        self.cluster_all = cluster_all
        self.n_jobs = n_jobs
        self.max_iter = max_iter
        
        seeds_arr = np.ascontiguousarray(np.asarray(seeds, dtype=np.float64)) if seeds is not None else None
        
        self._model = _core.MeanShift(
            bandwidth=bandwidth,
            seeds=seeds_arr,
            bin_seeding=bin_seeding,
            min_bin_freq=min_bin_freq,
            cluster_all=cluster_all,
            n_jobs=n_jobs,
            max_iter=max_iter
        )

    @_catch_panic
    def fit(self, X, y=None):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        self._model.fit(X)
        return self

    @_catch_panic
    def predict(self, X):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        return self._model.predict(X)

    @_catch_panic
    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.predict(X)
