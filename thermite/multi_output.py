import numpy as np
from typing import Any
from . import _core

class MultiOutputRegressor:
    """
    Multi-target regression.
    
    This strategy consists of fitting one regressor per target. This is a simple
    strategy for extending regressors that do not natively support multi-target
    regression.
    """
    def __init__(self, estimator: Any):
        self.estimator = estimator
        self._model = _core.MultiOutputRegressor(self.estimator)
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        self._model.fit(X, y)
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return self._model.predict(X)

    @property
    def estimators_(self):
        return self._model.estimators
