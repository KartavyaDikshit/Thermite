import numpy as np
from . import _core

class PCA:
    def __init__(self, n_components=2, *, random_state=None):
        self._model = _core.PCA(
            n_components=n_components,
            random_state=random_state
        )

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        self._model.fit(X)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        return self._model.transform(X)

    def fit_transform(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        return self._model.fit_transform(X)

    def inverse_transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        return self._model.inverse_transform(X)

    @property
    def components_(self):
        return self._model.components_

    @property
    def explained_variance_(self):
        return self._model.explained_variance_

    @property
    def explained_variance_ratio_(self):
        return self._model.explained_variance_ratio_

    @property
    def mean_(self):
        return self._model.mean_
