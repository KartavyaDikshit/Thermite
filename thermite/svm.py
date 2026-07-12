import numpy as np
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


class SVC:
    """
    Support Vector Classification (SVC).
    """
    def __init__(
        self,
        *,
        C=1.0,
        kernel="rbf",
        degree=3,
        gamma="scale",
        coef0=0.0,
        probability=False,
        tol=1e-3,
        max_iter=1000
    ):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.probability = probability
        self.tol = tol
        self.max_iter = max_iter
        
        self._model = _core.SVC(
            C=float(C),
            kernel=str(kernel),
            degree=int(degree),
            gamma=str(gamma),
            coef0=float(coef0),
            probability=bool(probability),
            eps=float(tol),
            max_iter=int(max_iter)
        )

    @_catch_panic
    def fit(self, X, y):
        """
        Fit the SVM model according to the given training data.
        """
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
        """
        Perform classification on samples in X.
        """
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        return self._model.predict(X)

    @_catch_panic
    def predict_proba(self, X):
        """
        Compute probabilities of possible outcomes for samples in X.
        """
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        return self._model.predict_proba(X)

    @property
    def classes_(self):
        return self._model.classes_
