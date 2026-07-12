
import numpy as np
from . import _core

class RFE:
    def __init__(self, estimator=None, n_features_to_select=5, step=1):
        self.n_features_to_select = n_features_to_select
        self.step = step
        self._model = _core.RFE(n_features_to_select=n_features_to_select, step=step)

    def fit(self, X, y):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        y = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
        self._model.fit(X, y)
        return self

    def transform(self, X):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        return self._model.transform(X)

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

    @property
    def support_(self):
        return self._model.support_

    @property
    def ranking_(self):
        return self._model.ranking_
