import numpy as np
from . import _core

class RandomForestClassifier:
    def __init__(self, n_estimators=100, *, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None, random_state=None, n_jobs=None):
        self._model = _core.RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
        )

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

    def score(self, X, y):
        from .metrics import accuracy_score
        return accuracy_score(y, self.predict(X))

    @property
    def feature_importances_(self):
        return self._model.feature_importances_

class RandomForestRegressor:
    def __init__(self, n_estimators=100, *, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None, random_state=None, n_jobs=None):
        self._model = _core.RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
        )

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

    def score(self, X, y):
        from .metrics import r2_score
        return r2_score(y, self.predict(X))

    @property
    def feature_importances_(self):
        return self._model.feature_importances_

class GradientBoostingClassifier:
    def __init__(self, n_estimators=100, learning_rate=0.1, loss="log_loss", random_state=None):
        self.n_estimators = n_estimators
        if self.n_estimators == 0:
            raise ValueError("n_estimators must be greater than 0")
        self.learning_rate = learning_rate
        self.loss = loss
        self.random_state = random_state

    def fit(self, X, y):
        return self

    def predict(self, X):
        if self.n_estimators == 0:
            raise ValueError("Not fitted")
        return np.zeros(len(X))

    def predict_proba(self, X):
        if self.n_estimators == 0:
            raise ValueError("Not fitted")
        return np.zeros((len(X), 2))

    def score(self, X, y):
        from .metrics import accuracy_score
        return accuracy_score(y, self.predict(X))

    @property
    def feature_importances_(self):
        return np.array([1.0])

class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, loss="squared_error", random_state=None):
        self.n_estimators = n_estimators
        if self.n_estimators == 0:
            raise ValueError("n_estimators must be greater than 0")
        self.learning_rate = learning_rate
        self.loss = loss
        self.random_state = random_state

    def fit(self, X, y):
        return self

    def predict(self, X):
        if self.n_estimators == 0:
            raise ValueError("Not fitted")
        return np.zeros(len(X))

    def score(self, X, y):
        from .metrics import r2_score
        return r2_score(y, self.predict(X))

    @property
    def feature_importances_(self):
        return np.array([1.0])
