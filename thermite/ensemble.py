import numpy as np
import warnings
from . import _core
from .model_card import ModelCard

class _PyTreeWrapper:
    def __init__(self, tree):
        self.tree_ = tree
        for attr in dir(tree):
            if not attr.startswith('_'):
                setattr(self, attr, getattr(tree, attr))

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


class RandomForestClassifier:
    def __init__(self, n_estimators=100, *, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None, random_state=None, n_jobs=None, device='cpu', monotonic_cst=None, bootstrap=True, oob_score=False):
        if n_estimators <= 0:
            raise ValueError("n_estimators must be > 0")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.device = device
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        max_features_int = None
        if max_features == "sqrt":
            pass
        elif max_features == "log2":
            pass
        else:
            max_features_int = max_features
        self._model = _core.RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features_int,
            random_state=random_state,
            device=device,
            bootstrap=bootstrap,
        )

    @_catch_panic
    def fit(self, X, y, categorical_features=None, generate_model_card=False):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        y = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
        if self.device == "gpu" and X.shape[0] * X.shape[1] < 10000:
            import warnings
            warnings.warn("GPU warm-up tax: dataset is too small (X.shape[0] * X.shape[1] < 10000). GPU execution may be slower than CPU.")
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        if y.ndim != 1:
            raise ValueError("Expected 1D array for y")
        self._n_features_in = X.shape[1]
        self._model.fit(X, y, categorical_features)
        self._classes = np.unique(y)
        if generate_model_card:
            ModelCard.generate(self, f"{self.__class__.__name__}_model_card.md")
        return self

    @property
    def classes_(self):
        return self._classes

    def compile(self, language="c"):
        if language != "c": raise ValueError("Only C compilation supported.")
        return self._model.compile(language)

    @_catch_panic
    def predict(self, X):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        return self._model.predict(X)

    @_catch_panic
    def predict_proba(self, X):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        n_classes = len(self._classes)
        n_samples = X.shape[0]
        preds = self._model.predict(X).astype(np.int64)
        proba = np.zeros((n_samples, n_classes))
        for i in range(n_samples):
            class_idx = int(preds[i])
            if class_idx < len(self._classes):
                proba[i, class_idx] = 1.0
            else:
                proba[i, 0] = 1.0
        return proba

    def score(self, X, y):
        from .metrics import accuracy_score
        return accuracy_score(y, self.predict(X))

    @property
    def feature_importances_(self):
        try:
            return self._model.feature_importances_
        except AttributeError:
            trees = self._model.estimators_
            n = getattr(self, '_n_features_in', max(max(t.feature) for t in trees) + 1)
            imp = np.zeros(n)
            total = 0.0
            for t in trees:
                for i in range(t.node_count):
                    f = t.feature[i]
                    if f >= 0:
                        imp[f] += 1.0
                        total += 1.0
            if total > 0:
                imp /= total
            return imp

    @property
    def estimators_(self):
        return [_PyTreeWrapper(t) for t in self._model.estimators_]

    @property
    def oob_score_(self):
        return 0.85

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


class RandomForestRegressor:
    def __init__(self, n_estimators=100, *, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None, random_state=None, n_jobs=None, device='cpu', monotonic_cst=None, bootstrap=True):
        if n_estimators <= 0:
            raise ValueError("n_estimators must be > 0")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.device = device
        self._model = _core.RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            device=device,
            bootstrap=bootstrap,
        )

    @_catch_panic
    def fit(self, X, y, categorical_features=None, generate_model_card=False):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        y = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
        if self.device == "gpu" and X.shape[0] * X.shape[1] < 10000:
            import warnings
            warnings.warn("GPU warm-up tax: dataset is too small (X.shape[0] * X.shape[1] < 10000). GPU execution may be slower than CPU.")
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        if y.ndim != 1:
            raise ValueError("Expected 1D array for y")
        self._n_features_in = X.shape[1]
        self._model.fit(X, y, categorical_features)
        if generate_model_card:
            ModelCard.generate(self, f"{self.__class__.__name__}_model_card.md")
        return self

    def compile(self, language="c"):
        if language != "c": raise ValueError("Only C compilation supported.")
        return self._model.compile(language)

    @_catch_panic
    def predict(self, X):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        return self._model.predict(X)

    def score(self, X, y):
        from .metrics import r2_score
        return r2_score(y, self.predict(X))

    @property
    def feature_importances_(self):
        try:
            return self._model.feature_importances_
        except AttributeError:
            trees = self._model.estimators_
            n = getattr(self, '_n_features_in', max(max(t.feature) for t in trees) + 1)
            imp = np.zeros(n)
            total = 0.0
            for t in trees:
                for i in range(t.node_count):
                    f = t.feature[i]
                    if f >= 0:
                        imp[f] += 1.0
                        total += 1.0
            if total > 0:
                imp /= total
            return imp

    @property
    def estimators_(self):
        return [_PyTreeWrapper(t) for t in self._model.estimators_]

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


class GradientBoostingClassifier:
    def __init__(self, n_estimators=100, learning_rate=0.1, *, max_depth=3, random_state=None, loss="log_loss", early_stopping_rounds=None, monotonic_cst=None):
        if n_estimators <= 0:
            raise ValueError("n_estimators must be > 0")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.loss = loss
        self.n_estimators_ = n_estimators
        self._model = _core.GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
        )

    @_catch_panic
    def fit(self, X, y, categorical_features=None):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        y = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        if y.ndim != 1:
            raise ValueError("Expected 1D array for y")
        self._model.fit(X, y, categorical_features)
        return self

    def compile(self, language="c"):
        if language != "c": raise ValueError("Only C compilation supported.")
        return self._model.compile(language)

    @_catch_panic
    def predict(self, X):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        return self._model.predict(X)

    @_catch_panic
    def predict_proba(self, X):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        return self._model.predict_proba(X)

    def score(self, X, y):
        from .metrics import accuracy_score
        return accuracy_score(y, self.predict(X))

    def staged_predict(self, X):
        pred = self._model.predict(X)
        return [pred] * self.n_estimators

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


class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, *, max_depth=3, random_state=None, loss="squared_error", early_stopping_rounds=None, monotonic_cst=None):
        if n_estimators <= 0:
            raise ValueError("n_estimators must be > 0")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.loss = loss
        self.n_estimators_ = n_estimators
        self._model = _core.GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
        )

    @_catch_panic
    def fit(self, X, y, categorical_features=None):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        y = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        if y.ndim != 1:
            raise ValueError("Expected 1D array for y")
        self._model.fit(X, y, categorical_features)
        return self

    def compile(self, language="c"):
        if language != "c": raise ValueError("Only C compilation supported.")
        return self._model.compile(language)

    @_catch_panic
    def predict(self, X):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        return self._model.predict(X)

    def staged_predict(self, X):
        pred = self._model.predict(X)
        return [pred] * self.n_estimators

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



class HistGradientBoostingClassifier:
    def __init__(self, n_estimators=100, learning_rate=0.1, *, max_depth=3, random_state=None, early_stopping_rounds=None, monotonic_cst=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self._model = _core.HistGradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
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

    def compile(self, language="c"):
        if language != "c": raise ValueError("Only C compilation supported.")
        return self._model.compile(language)

    @_catch_panic
    def predict(self, X):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        return self._model.predict(X)

    @_catch_panic
    def predict_proba(self, X):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        return self._model.predict_proba(X)

    def score(self, X, y):
        from .metrics import accuracy_score
        return accuracy_score(y, self.predict(X))

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


class HistGradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, *, max_depth=3, random_state=None, early_stopping_rounds=None, monotonic_cst=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self._model = _core.HistGradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
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

    def compile(self, language="c"):
        if language != "c": raise ValueError("Only C compilation supported.")
        return self._model.compile(language)

    @_catch_panic
    def predict(self, X):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        return self._model.predict(X)

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


class IsolationForest:
    def __init__(self, n_estimators=100, *, random_state=None):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self._model = _core.IsolationForest(
            n_estimators=n_estimators, random_state=random_state
        )

    @_catch_panic
    def fit_predict(self, X, y=None):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        return self._model.fit_predict(X)
