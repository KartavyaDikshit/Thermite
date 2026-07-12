
import numpy as np
from . import _core

class AutoRegressive:
    def __init__(self, lags=1):
        self.lags = lags
        self._model = _core.AutoRegressive(lags=lags)

    def fit(self, y):
        y = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
        self._model.fit(y)
        return self

    def predict(self, steps, last_y):
        last_y = np.ascontiguousarray(np.asarray(last_y, dtype=np.float64))
        return self._model.predict(steps, last_y)
