import numpy as np
from . import _core

class KNeighborsClassifier:
    def __init__(self, n_neighbors=5, *, weights="uniform"):
        self._model = _core.KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights
        )

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        if y.ndim != 1:
            raise ValueError("Expected 1D array for y")
        self._model.fit(X, y)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        return self._model.predict(X)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        return self._model.predict_proba(X)
