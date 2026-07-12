
import numpy as np
from . import _core

class SurvivalForest:
    def __init__(self, n_estimators=100):
        self.n_estimators = n_estimators
        self._model = _core.SurvivalForest(n_estimators=n_estimators)

    def fit(self, X, times, events):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        times = np.ascontiguousarray(np.asarray(times, dtype=np.float64))
        events = np.ascontiguousarray(np.asarray(events, dtype=np.float64))
        self._model.fit(X, times, events)
        return self

    def predict_survival_function(self, X, times_to_predict):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        times_to_predict = np.ascontiguousarray(np.asarray(times_to_predict, dtype=np.float64))
        return self._model.predict_survival_function(X, times_to_predict)
