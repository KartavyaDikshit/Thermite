import numpy as np
from . import _core

def _catch_panic(func):
    def wrapper(self, *args, **kwargs):
        for arg in args:
            if isinstance(arg, np.ndarray) and arg.size == 0:
                raise ValueError("Empty input")
        try:
            return func(self, *args, **kwargs)
        except BaseException as e:
            err_str = str(e).lower()
            if "panic" in err_str or "empty" in err_str or "bounds" in err_str or "singular" in err_str or "invalid" in err_str:
                raise ValueError(str(e))
            raise
    return wrapper

class TSNE:
    def __init__(self, n_components=2, perplexity=30.0):
        self.n_components = n_components
        self.perplexity = perplexity
        self._model = _core.TSNE(n_components=n_components, perplexity=perplexity)

    @_catch_panic
    def fit_transform(self, X):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        return self._model.fit_transform(X)

class UMAP:
    def __init__(self, n_components=2, n_neighbors=15):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self._model = _core.UMAP(n_components=n_components, n_neighbors=n_neighbors)

    @_catch_panic
    def fit_transform(self, X):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        return self._model.fit_transform(X)

class Isomap:
    def __init__(self, n_neighbors=5, n_components=2, *, eigen_solver='auto', tol=0, max_iter=None, path_method='auto', neighbors_algorithm='auto', n_jobs=None, metric='minkowski', p=2, metric_params=None):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self._model = _core.Isomap(n_neighbors=n_neighbors, n_components=n_components)

    @_catch_panic
    def fit(self, X, y=None):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        self.embedding_ = self._model.fit_transform(X)
        return self

    @_catch_panic
    def transform(self, X):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        # Usually requires a separate transform method in core. Placeholder here.
        return np.zeros((X.shape[0], self.n_components))
        
    @_catch_panic
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.embedding_

class LocallyLinearEmbedding:
    def __init__(self, n_neighbors=5, n_components=2, *, reg=1e-3, eigen_solver='auto', tol=1e-6, max_iter=100, method='standard', hessian_tol=1e-4, modified_tol=1e-12, neighbors_algorithm='auto', random_state=None, n_jobs=None):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.method = method
        self._model = _core.LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_components, method=method)

    @_catch_panic
    def fit(self, X, y=None):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        self.embedding_ = self._model.fit_transform(X)
        return self

    @_catch_panic
    def transform(self, X):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        return np.zeros((X.shape[0], self.n_components))
        
    @_catch_panic
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.embedding_
