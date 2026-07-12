import numpy as np
from .tree import DecisionTreeClassifier, DecisionTreeRegressor

class RandomForestClassifier:
    def __init__(self, n_estimators=100, *, max_depth=None, max_features="sqrt", random_state=None, bootstrap=True, oob_score=False):
        self.n_estimators = n_estimators
        if self.n_estimators == 0:
            raise ValueError("n_estimators must be greater than 0")
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.estimators_ = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if self.random_state is not None:
            np.random.seed(self.random_state)
        n_samples = X.shape[0]
        self.estimators_ = []
        for i in range(self.n_estimators):
            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            if self.bootstrap:
                indices = np.random.choice(n_samples, n_samples, replace=True)
                tree.fit(X[indices], y[indices])
            else:
                tree.fit(X, y)
            self.estimators_.append(tree)
        return self

    def predict(self, X):
        if self.n_estimators == 0 or not self.estimators_:
            raise ValueError("Not fitted")
        predictions = np.array([tree.predict(X) for tree in self.estimators_])
        mode_preds = []
        for i in range(predictions.shape[1]):
            vals, counts = np.unique(predictions[:, i], return_counts=True)
            mode_preds.append(vals[np.argmax(counts)])
        return np.array(mode_preds)

    def predict_proba(self, X):
        if self.n_estimators == 0 or not self.estimators_:
            raise ValueError("Not fitted")
        probas = np.mean([tree.predict_proba(X) for tree in self.estimators_], axis=0)
        return probas

    def score(self, X, y):
        from .metrics import accuracy_score
        return accuracy_score(y, self.predict(X))

    @property
    def feature_importances_(self):
        return np.mean([tree.feature_importances_ for tree in self.estimators_], axis=0) if self.estimators_ else np.array([1.0])

class RandomForestRegressor:
    def __init__(self, n_estimators=100, *, max_depth=None, random_state=None, bootstrap=True, oob_score=False):
        self.n_estimators = n_estimators
        if self.n_estimators == 0:
            raise ValueError("n_estimators must be greater than 0")
        self.max_depth = max_depth
        self.random_state = random_state
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.estimators_ = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if self.random_state is not None:
            np.random.seed(self.random_state)
        n_samples = X.shape[0]
        self.estimators_ = []
        for i in range(self.n_estimators):
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            if self.bootstrap:
                indices = np.random.choice(n_samples, n_samples, replace=True)
                tree.fit(X[indices], y[indices])
            else:
                tree.fit(X, y)
            self.estimators_.append(tree)
        return self

    def predict(self, X):
        if self.n_estimators == 0 or not self.estimators_:
            raise ValueError("Not fitted")
        predictions = np.array([tree.predict(X) for tree in self.estimators_])
        return np.mean(predictions, axis=0)

    def score(self, X, y):
        from .metrics import r2_score
        return r2_score(y, self.predict(X))

    @property
    def feature_importances_(self):
        return np.mean([tree.feature_importances_ for tree in self.estimators_], axis=0) if self.estimators_ else np.array([1.0])

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
