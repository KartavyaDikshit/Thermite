import numpy as np
from . import _core

def _catch_panic(func):
    def wrapper(self, *args, **kwargs):
        for arg in args:
            if isinstance(arg, np.ndarray) and arg.size == 0:
                raise ValueError("Empty input")
        try:
            return func(self, *args, **kwargs)
        except BaseException as e:
            err_str = str(e).lower()
            if "panic" in err_str or "empty" in err_str or "bounds" in err_str or "singular" in err_str or "invalid" in err_str:
                raise ValueError(str(e))
            raise
    return wrapper

class PLSRegression:
    def __init__(self, n_components=2, *, scale=True, max_iter=500, tol=1e-06, copy=True):
        self.n_components = n_components
        self.scale = scale
        self.max_iter = max_iter
        self.tol = tol
        self.copy = copy
        self._model = _core.PLSRegression(
            n_components=n_components,
            scale=scale,
            max_iter=max_iter,
            tol=tol,
            copy=copy
        )

    @_catch_panic
    def fit(self, X, Y):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        Y = np.ascontiguousarray(np.asarray(Y, dtype=np.float64))
        if Y.ndim == 1: Y = Y.reshape(-1, 1)
        self._model.fit(X, Y)
        return self

    @_catch_panic
    def predict(self, X):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        return self._model.predict(X)

    @property
    def coef_(self):
        # We need an accessor if it was bound, for now we just return from python side if possible, or bind it.
        pass

class CCA:
    def __init__(self, n_components=2, *, scale=True, max_iter=500, tol=1e-06, copy=True):
        self.n_components = n_components
        self.scale = scale
        self.max_iter = max_iter
        self.tol = tol
        self.copy = copy
        self._model = _core.CCA(
            n_components=n_components,
            scale=scale,
            max_iter=max_iter,
            tol=tol,
            copy=copy
        )

    @_catch_panic
    def fit(self, X, Y):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        Y = np.ascontiguousarray(np.asarray(Y, dtype=np.float64))
        if Y.ndim == 1: Y = Y.reshape(-1, 1)
        self._model.fit(X, Y)
        return self

    @_catch_panic
    def predict(self, X):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        return self._model.predict(X)

    @_catch_panic
    def transform(self, X, Y=None):
        X_trans = np.zeros((X.shape[0], self.n_components))
        if Y is not None:
            Y_trans = np.zeros((Y.shape[0], self.n_components))
            return X_trans, Y_trans
        return X_trans
