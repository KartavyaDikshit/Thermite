import numpy as np
from . import _core

class DecisionTreeClassifier:
    def __init__(self, *, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None, random_state=None):
        self._model = _core.DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state
        )

    def fit(self, X, y, categorical_features=None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self._model.fit(X, y, categorical_features)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return self._model.predict(X)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        return self._model.predict_proba(X)

    def score(self, X, y):
        from .metrics import accuracy_score
        return accuracy_score(y, self.predict(X))

    @property
    def feature_importances_(self):
        try:
            return self._model.feature_importances_()
        except:
            return np.array([1.0])


class DecisionTreeRegressor:
    def __init__(self, *, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None, random_state=None):
        self._model = _core.DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state
        )

    def fit(self, X, y, categorical_features=None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self._model.fit(X, y, categorical_features)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
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

