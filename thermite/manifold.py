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
