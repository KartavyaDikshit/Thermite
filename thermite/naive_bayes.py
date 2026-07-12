import numpy as np
from . import _core

class GaussianNB:
    """
    Gaussian Naive Bayes.
    """
    def __init__(self):
        self._model = _core.GaussianNB()

    def fit(self, X, y):
        """
        Fit Gaussian Naive Bayes according to X, y.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        if y.ndim != 1:
            raise ValueError("Expected 1D array for y")
        
        self._model.fit(X, y)
        return self

    def partial_fit(self, X, y, classes=None):
        """
        Incremental fit on a batch of samples.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        if y.ndim != 1:
            raise ValueError("Expected 1D array for y")
        
        self._model.partial_fit(X, y, classes)
        return self

    def predict(self, X):
        """
        Perform classification on an array of test vectors X.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        return self._model.predict(X)

    def predict_proba(self, X):
        """
        Return probability estimates for the test vector X.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        return self._model.predict_proba(X)
