import numpy as np
from . import _core

def _catch_panic(func):
    def wrapper(self, *args, **kwargs):
        for arg in args:
            if isinstance(arg, np.ndarray) and arg.size == 0:
                raise ValueError("Empty input")
            if isinstance(arg, (list, tuple)) and len(arg) == 0:
                raise ValueError("Empty input")
        
        try:
            return func(self, *args, **kwargs)
        except BaseException as e:
            err_str = str(e).lower()
            if "panic" in err_str or "empty" in err_str or "bounds" in err_str or "singular" in err_str or "invalid" in err_str:
                raise ValueError(str(e))
            raise
    return wrapper

class GaussianMixture:
    def __init__(self, n_components=1, covariance_type='full', tol=1e-3, reg_covar=1e-6, max_iter=100, n_init=1, init_params='kmeans', weights_init=None, means_init=None, precisions_init=None, random_state=None, warm_start=False, verbose=0, verbose_interval=10):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.max_iter = max_iter
        self._model = _core.GaussianMixture(
            n_components=n_components,
            max_iter=max_iter,
            tol=tol
        )

    @_catch_panic
    def fit(self, X, y=None):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        self._model.fit(X)
        return self

    @_catch_panic
    def predict(self, X):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        return self._model.predict(X)
    
    @_catch_panic
    def predict_proba(self, X):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        return self._model.predict_proba(X)
