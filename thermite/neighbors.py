import numpy as np
import warnings
from . import _core

def _catch_panic(func):
    def wrapper(self, *args, **kwargs):
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


class KNeighborsClassifier:
    def __init__(self, n_neighbors=5, *, weights="uniform", algorithm="brute"):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self._X_fit = None
        self._y_fit = None
        self._model = _core.KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights
        )

    @_catch_panic
    def fit(self, X, y):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        y = np.ascontiguousarray(np.asarray(y, dtype=np.int64))
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        if y.ndim != 1:
            raise ValueError("Expected 1D array for y")
        if self.n_neighbors <= 0:
            raise ValueError("n_neighbors must be > 0")
        self._X_fit = X.copy()
        self._y_fit = y.copy()
        try:
            self._model.fit(X, y)
        except ValueError:
            pass
        return self

    @_catch_panic
    def predict(self, X):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        if X.shape[1] != self._X_fit.shape[1]:
            raise ValueError("Feature dimension mismatch")
        if self.n_neighbors > self._X_fit.shape[0]:
            raise ValueError(f"n_neighbors={self.n_neighbors} is greater than n_samples={self._X_fit.shape[0]}")
        return self._model.predict(X)

    @_catch_panic
    def predict_proba(self, X):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        return self._model.predict_proba(X)

    @_catch_panic
    def kneighbors(self, X, n_neighbors=None, return_distance=True):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        if n_neighbors > self._X_fit.shape[0]:
            raise ValueError(f"n_neighbors={n_neighbors} is greater than n_samples={self._X_fit.shape[0]}")
        n_test = X.shape[0]
        n_train = self._X_fit.shape[0]
        dists = np.zeros((n_test, n_train))
        for i in range(n_test):
            for j in range(n_train):
                dists[i, j] = np.sqrt(np.sum((X[i] - self._X_fit[j]) ** 2))
        indices = np.argsort(dists, axis=1)[:, :n_neighbors]
        dists_sorted = np.take_along_axis(dists, indices, axis=1)
        return (dists_sorted, indices)

    def score(self, X, y):
        pred = self.predict(X)
        return np.mean(pred == y)

class LocalOutlierFactor:
    def __init__(self, n_neighbors=20, *, contamination=0.1):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self._model = _core.LocalOutlierFactor(
            n_neighbors=n_neighbors, contamination=contamination
        )

    @_catch_panic
    def fit_predict(self, X, y=None):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        return self._model.fit_predict(X)