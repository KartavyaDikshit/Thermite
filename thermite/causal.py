from ._core import TLearner as CoreTLearner
import numpy as np

class TLearner:
    """T-Learner for conditional average treatment effect (CATE) estimation.
    """
    def __init__(self):
        self._core = CoreTLearner()

    def fit(self, X: np.ndarray, treatment: np.ndarray, y: np.ndarray):
        X_arr = np.ascontiguousarray(X, dtype=np.float64)
        t_arr = np.ascontiguousarray(treatment, dtype=np.float64)
        y_arr = np.ascontiguousarray(y, dtype=np.float64)
        self._core.fit(X_arr, t_arr, y_arr)
        return self

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.ascontiguousarray(X, dtype=np.float64)
        return self._core.predict_cate(X_arr)
