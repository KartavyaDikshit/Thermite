import numpy as np
from . import _core

class LinearRegression:
    def __init__(self, *, fit_intercept=True):
        self._model = _core.LinearRegression(fit_intercept=fit_intercept)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
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

    @property
    def coef_(self):
        return self._model.coef_

    @property
    def intercept_(self):
        return self._model.intercept_


class Ridge:
    def __init__(self, alpha=1.0, *, fit_intercept=True):
        self._model = _core.Ridge(alpha=alpha, fit_intercept=fit_intercept)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
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

    @property
    def coef_(self):
        return self._model.coef_

    @property
    def intercept_(self):
        return self._model.intercept_


class Lasso:
    def __init__(self, alpha=1.0, *, fit_intercept=True, max_iter=1000, tol=1e-4):
        self._model = _core.Lasso(alpha=alpha, fit_intercept=fit_intercept, max_iter=max_iter, tol=tol)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
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

    @property
    def coef_(self):
        return self._model.coef_

    @property
    def intercept_(self):
        return self._model.intercept_


class LogisticRegression:
    def __init__(self, penalty='l2', *, C=1.0, tol=1e-4, max_iter=100):
        self._model = _core.LogisticRegression(C=C, max_iter=max_iter, tol=tol, penalty=penalty)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
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

    @property
    def coef_(self):
        return self._model.coef_

    @property
    def intercept_(self):
        return self._model.intercept_

    @property
    def classes_(self):
        return self._model.classes_

class LinearSVC:
    def __init__(self, penalty='l2', loss='squared_hinge', *, dual=True, tol=1e-4, C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=1000):
        # Using subset of arguments internally, just accepting sklearn signature
        self._model = _core.LinearSVC(C=C, max_iter=max_iter, tol=tol)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
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

    @property
    def coef_(self):
        return self._model.coef_

    @property
    def intercept_(self):
        return self._model.intercept_

    @property
    def classes_(self):
        return self._model.classes_
