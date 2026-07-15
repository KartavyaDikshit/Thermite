import numpy as np
import warnings
from . import _core

def _catch_panic(func):
    def wrapper(self, *args, **kwargs):
        # basic input validation for all estimators
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


class LinearRegression:
    def __init__(self, *, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self._model = _core.LinearRegression(fit_intercept=fit_intercept)

    @_catch_panic
    def fit(self, X, y, sample_weight=None):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        y = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()
        if y.ndim == 2:
            self._multi_output = True
            self._multi_models = []
            for i in range(y.shape[1]):
                m = _core.LinearRegression(fit_intercept=self.fit_intercept)
                m.fit(X, y[:, i])
                self._multi_models.append(m)
            return self
        if y.ndim != 1:
            raise ValueError("Expected 1D array for y")
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)
        self._model.fit(X, y)
        return self

    @_catch_panic
    def predict(self, X):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        if hasattr(self, '_multi_output') and self._multi_output:
            preds = np.column_stack([m.predict(X) for m in self._multi_models])
            return preds
        return self._model.predict(X)

    @property
    def coef_(self):
        if hasattr(self, '_multi_output') and self._multi_output:
            return np.column_stack([m.coef_ for m in self._multi_models])
        return self._model.coef_

    @property
    def intercept_(self):
        if hasattr(self, '_multi_output') and self._multi_output:
            return np.array([m.intercept_ for m in self._multi_models])
        return self._model.intercept_

    @_catch_panic
    def predict(self, X):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        return self._model.predict(X)

    @_catch_panic
    def to_onnx(self, filepath: str):
        self._model.to_onnx(filepath)

    def save_checkpoint(self, filepath: str):
        self._model.save_checkpoint(filepath)

    @classmethod
    def load_checkpoint(cls, filepath: str):
        import inspect
        sig = inspect.signature(cls.__init__)
        kwargs = {}
        for param in sig.parameters.values():
            if param.name != 'self' and param.default is not inspect.Parameter.empty:
                kwargs[param.name] = param.default
        obj = cls(**kwargs)
        obj._model = type(obj._model).load_checkpoint(filepath)
        return obj

    def score(self, X, y):
        from .metrics import r2_score
        return r2_score(y, self.predict(X))



class Ridge:
    def __init__(self, alpha=1.0, *, fit_intercept=True, solver='auto', random_state=None):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.random_state = random_state
        self._model = _core.Ridge(alpha=alpha, fit_intercept=fit_intercept)

    @_catch_panic
    def fit(self, X, y):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        y = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        if y.ndim != 1:
            raise ValueError("Expected 1D array for y")
    @_catch_panic
    def fit(self, X, y):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        y = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        if y.ndim != 1:
            raise ValueError("Expected 1D array for y")
        self._model.fit(X, y)
        return self

    @_catch_panic
    def predict(self, X):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        return self._model.predict(X)

    @property
    def coef_(self):
        return self._model.coef_

    @property
    def intercept_(self):
        return self._model.intercept_

    def score(self, X, y):
        from .metrics import r2_score
        return r2_score(y, self.predict(X))

    def save_checkpoint(self, filepath: str):
        self._model.save_checkpoint(filepath)

    @classmethod
    def load_checkpoint(cls, filepath: str):
        import inspect
        sig = inspect.signature(cls.__init__)
        kwargs = {}
        for param in sig.parameters.values():
            if param.name != 'self' and param.default is not inspect.Parameter.empty:
                kwargs[param.name] = param.default
        obj = cls(**kwargs)
        obj._model = type(obj._model).load_checkpoint(filepath)
        return obj



class Lasso:
    def __init__(self, alpha=1.0, *, fit_intercept=True, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
    def __init__(self, alpha=1.0, *, fit_intercept=True, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self._model = _core.Lasso(alpha=alpha, fit_intercept=fit_intercept, max_iter=max_iter, tol=tol)

    @_catch_panic
    def fit(self, X, y):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        y = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        if y.ndim != 1:
            raise ValueError("Expected 1D array for y")
    @_catch_panic
    def fit(self, X, y):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        y = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        if y.ndim != 1:
            raise ValueError("Expected 1D array for y")
        self._model.fit(X, y)
        self.n_iter_ = self.max_iter
        return self

    @_catch_panic
    def predict(self, X):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        return self._model.predict(X)

    @property
    def coef_(self):
        return self._model.coef_

    @property
    def intercept_(self):
        return self._model.intercept_

    def score(self, X, y):
        from .metrics import r2_score
        return r2_score(y, self.predict(X))

    def save_checkpoint(self, filepath: str):
        self._model.save_checkpoint(filepath)

    @classmethod
    def load_checkpoint(cls, filepath: str):
        import inspect
        sig = inspect.signature(cls.__init__)
        kwargs = {}
        for param in sig.parameters.values():
            if param.name != 'self' and param.default is not inspect.Parameter.empty:
                kwargs[param.name] = param.default
        obj = cls(**kwargs)
        obj._model = type(obj._model).load_checkpoint(filepath)
        return obj



class LogisticRegression:
    def __init__(self, penalty='l2', *, C=1.0, tol=1e-4, max_iter=100, solver='lbfgs', random_state=None):
        self.penalty = penalty
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        self.solver = solver
        self.random_state = random_state
        self._model = _core.LogisticRegression(C=C, max_iter=max_iter, tol=tol, penalty=penalty)

    @_catch_panic
    def fit(self, X, y):
        import scipy.sparse
        y = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
        if y.ndim != 1:
            raise ValueError("Expected 1D array for y")
        if len(np.unique(y)) < 2:
            raise ValueError("LogisticRegression requires at least 2 classes")
            
        if scipy.sparse.issparse(X):
            X_csr = X.tocsr()
            self._model.fit_sparse(
                np.ascontiguousarray(X_csr.data.astype(np.float64)),
                np.ascontiguousarray(X_csr.indices.astype(np.uintp)),
                np.ascontiguousarray(X_csr.indptr.astype(np.uintp)),
                X_csr.shape[0],
                X_csr.shape[1],
                y
            )
        else:
            X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
            if X.ndim != 2:
                raise ValueError("Expected 2D array for X")
            self._model.fit(X, y)
        self._classes = np.unique(y)
        return self

    def partial_fit(self, X, y, classes=None):
        """Incremental fit with SGD - for streaming out-of-core learning."""
        y = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        if y.ndim != 1:
            raise ValueError("Expected 1D array for y")
        self._model.partial_fit(X, y, classes)
        return self

    @_catch_panic
    def predict(self, X):
        import scipy.sparse
        if scipy.sparse.issparse(X):
            X_csr = X.tocsr()
            return self._model.predict_sparse(
                np.ascontiguousarray(X_csr.data.astype(np.float64)),
                np.ascontiguousarray(X_csr.indices.astype(np.uintp)),
                np.ascontiguousarray(X_csr.indptr.astype(np.uintp)),
                X_csr.shape[0],
                X_csr.shape[1]
            )
        else:
            X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
            if X.ndim != 2:
                raise ValueError("Expected 2D array for X")
            return self._model.predict(X)

    @_catch_panic
    def predict_proba(self, X):
        import scipy.sparse
        if scipy.sparse.issparse(X):
            X_csr = X.tocsr()
            return self._model.predict_proba_sparse(
                np.ascontiguousarray(X_csr.data.astype(np.float64)),
                np.ascontiguousarray(X_csr.indices.astype(np.uintp)),
                np.ascontiguousarray(X_csr.indptr.astype(np.uintp)),
                X_csr.shape[0],
                X_csr.shape[1]
            )
        else:
            X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
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
        return self._classes

    def save_checkpoint(self, filepath: str):
        self._model.save_checkpoint(filepath)

    @classmethod
    def load_checkpoint(cls, filepath: str):
        import inspect
        sig = inspect.signature(cls.__init__)
        kwargs = {}
        for param in sig.parameters.values():
            if param.name != 'self' and param.default is not inspect.Parameter.empty:
                kwargs[param.name] = param.default
        obj = cls(**kwargs)
        obj._model = type(obj._model).load_checkpoint(filepath)
        return obj


class LinearSVC:
    def __init__(self, penalty='l2', loss='squared_hinge', *, dual=True, tol=1e-4, C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=1000):
        self.penalty = penalty
        self.loss = loss
        self.dual = dual
        self.tol = tol
        self.C = C
        self.multi_class = multi_class
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.verbose = verbose
        self.random_state = random_state
        self.max_iter = max_iter
        # Using subset of arguments internally, just accepting sklearn signature
        self._model = _core.LinearSVC(C=C, max_iter=max_iter, tol=tol)

    @_catch_panic
    def fit(self, X, y):
        import scipy.sparse
        y = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
        if y.ndim != 1:
            raise ValueError("Expected 1D array for y")
            
        if scipy.sparse.issparse(X):
            X_csr = X.tocsr()
            self._model.fit_sparse(
                np.ascontiguousarray(X_csr.data.astype(np.float64)),
                np.ascontiguousarray(X_csr.indices.astype(np.uintp)),
                np.ascontiguousarray(X_csr.indptr.astype(np.uintp)),
                X_csr.shape[0],
                X_csr.shape[1],
                y
            )
        else:
            X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
            if X.ndim != 2:
                raise ValueError("Expected 2D array for X")
            self._model.fit(X, y)
        return self

    @_catch_panic
    def predict(self, X):
        import scipy.sparse
        if scipy.sparse.issparse(X):
            X_csr = X.tocsr()
            return self._model.predict_sparse(
                np.ascontiguousarray(X_csr.data.astype(np.float64)),
                np.ascontiguousarray(X_csr.indices.astype(np.uintp)),
                np.ascontiguousarray(X_csr.indptr.astype(np.uintp)),
                X_csr.shape[0],
                X_csr.shape[1]
            )
        else:
            X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
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

class SGDClassifier:
    def __init__(self, loss='log', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=1000, tol=1e-3, learning_rate=0.01):
        self.loss = loss
        self.penalty = penalty
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        self._model = _core.SGDClassifier(
            loss=loss, penalty=penalty, alpha=alpha, l1_ratio=l1_ratio,
            fit_intercept=fit_intercept, max_iter=max_iter, tol=tol, learning_rate=learning_rate
        )

    @_catch_panic
    def fit(self, X, y):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        y = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        if y.ndim != 1:
            raise ValueError("Expected 1D array for y")
        self._model.fit(X, y)
        return self

    @_catch_panic
    def partial_fit(self, X, y, classes=None):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        y = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        if y.ndim != 1:
            raise ValueError("Expected 1D array for y")
        self._model.partial_fit(X, y, classes)
        return self

    @_catch_panic
    def predict(self, X):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
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
