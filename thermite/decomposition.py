import numpy as np
import warnings
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


class PCA:
    def __init__(self, n_components=None, *, random_state=None):
        self.n_components = n_components
        self.random_state = random_state
        self._model = None

    @_catch_panic
    def fit(self, X, y=None):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        n_components = self.n_components
        if n_components is None:
            n_components = min(X.shape[0], X.shape[1])
        if n_components <= 0:
            raise ValueError("n_components must be >= 1")
        if n_components > min(X.shape[0], X.shape[1]):
            raise ValueError("n_components must be <= min(n_samples, n_features)")
        self._n_samples = X.shape[0]
        self._model = _core.PCA(
            n_components=n_components,
            random_state=self.random_state
        )
        self._model.fit(X)
        return self

    @_catch_panic
    def transform(self, X):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        return self._model.transform(X)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def inverse_transform(self, X):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        return self._model.inverse_transform(X)

    @property
    def components_(self):
        return self._model.components_

    @property
    def explained_variance_(self):
        ev = self._model.explained_variance_
        if ev is not None and self._n_samples == 1:
            ev = np.array([np.nan] * len(ev))
        return ev

    @property
    def explained_variance_ratio_(self):
        return self._model.explained_variance_ratio_

    @property
    def mean_(self):
        return self._model.mean_

    @property
    def singular_values_(self):
        ev = self.explained_variance_
        if ev is None:
            return None
        n_samples = getattr(self, '_n_samples', None)
        if n_samples is None:
            return None
        return np.sqrt(ev * (n_samples - 1))

class DictionaryLearning:
    def __init__(self, n_components=None, *, alpha=1, max_iter=1000, tol=1e-8, fit_algorithm='lars', transform_algorithm='omp', transform_n_nonzero_coefs=None, transform_alpha=None, n_jobs=None, code_init=None, dict_init=None, verbose=False, split_sign=False, random_state=None, positive_code=False, positive_dict=False, transform_max_iter=1000):
        self.n_components = n_components
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.fit_algorithm = fit_algorithm
        self.transform_algorithm = transform_algorithm
        self.transform_alpha = transform_alpha
        self.n_jobs = n_jobs
        self.random_state = random_state
        self._model = _core.DictionaryLearning(
            n_components=n_components,
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
            fit_algorithm=fit_algorithm,
            transform_algorithm=transform_algorithm,
            transform_alpha=transform_alpha,
            n_jobs=n_jobs,
            random_state=random_state
        )

    @_catch_panic
    def fit(self, X, y=None):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        self._model.fit(X)
        return self

    @_catch_panic
    def transform(self, X):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        return self._model.transform(X)

    @_catch_panic
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    @property
    def components_(self):
        return self._model.components_