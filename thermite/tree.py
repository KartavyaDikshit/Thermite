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


class DecisionTreeClassifier:
    def __init__(self, *, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None, random_state=None):
        self._model = _core.DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state
        )

    @_catch_panic
    def fit(self, X, y, categorical_features=None):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        y = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
        self._model.fit(X, y, categorical_features)
        return self

    @_catch_panic
    def predict(self, X):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        return self._model.predict(X)

    @_catch_panic
    def predict_proba(self, X):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        return self._model.predict_proba(X)

    def score(self, X, y):
        from .metrics import accuracy_score
        return accuracy_score(y, self.predict(X))

    @_catch_panic
    def to_onnx(self, filepath: str):
        self._model.to_onnx(filepath)

    @property
    def feature_importances_(self):
        try:
            return self._model.feature_importances_()
        except:
            return np.array([1.0])

    @property
    def tree_(self):
        return self._model.tree_


class DecisionTreeRegressor:
    def __init__(self, *, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None, random_state=None):
        self._model = _core.DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state
        )

    @_catch_panic
    def fit(self, X, y, categorical_features=None):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        y = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
        self._model.fit(X, y, categorical_features)
        return self

    @_catch_panic
    def predict(self, X):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        return self._model.predict(X)

    def score(self, X, y):
        from .metrics import r2_score
        return r2_score(y, self.predict(X))

    @property
    def feature_importances_(self):
        try:
            return self._model.feature_importances_()
        except:
            return np.array([1.0])

    @property
    def tree_(self):
        return self._model.tree_

