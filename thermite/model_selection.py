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


def train_test_split(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None):
    """Split arrays or matrices into random train and test subsets.
    
    Compatible with scikit-learn.
    """
    if len(arrays) == 0:
        raise ValueError("At least one array is required as input")
        
    np_arrays = [np.asarray(arr) for arr in arrays]
    
    if stratify is not None:
        stratify = np.asarray(stratify)
        
    return _core.train_test_split(
        *np_arrays,
        test_size=test_size,
        train_size=train_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=stratify
    )

import itertools
import copy
from concurrent.futures import ThreadPoolExecutor

class KFold:
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        indices = np.arange(n_samples)
        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            rng.shuffle(indices)

        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[:n_samples % self.n_splits] += 1
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_indices = indices[start:stop]
            train_indices = np.concatenate([indices[:start], indices[stop:]])
            yield train_indices, test_indices
            current = stop

class StratifiedKFold:
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self._core = _core.StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def split(self, X, y, groups=None):
        y = np.asarray(y, dtype=np.int64)
        for train, test in self._core.split(X, y):
            yield train, test

class TimeSeriesSplit:
    def __init__(self, n_splits=5):
        self.n_splits = n_splits
        self._core = _core.TimeSeriesSplit(n_splits=n_splits)

    def split(self, X, y=None, groups=None):
        for train, test in self._core.split(X):
            yield train, test

class GroupKFold:
    def __init__(self, n_splits=5):
        self.n_splits = n_splits
        self._core = _core.GroupKFold(n_splits=n_splits)

    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        groups = np.asarray(groups, dtype=np.int64)
        for train, test in self._core.split(X, y, groups):
            yield train, test


_SCORERS = {
    "accuracy": lambda m, X, y: np.mean(m.predict(X) == y),
    "f1": lambda m, X, y: _f1_score(y, m.predict(X)),
    "precision": lambda m, X, y: _precision_score(y, m.predict(X)),
    "recall": lambda m, X, y: _recall_score(y, m.predict(X)),
    "neg_mean_squared_error": lambda m, X, y: -np.mean((m.predict(X) - y) ** 2),
    "r2": lambda m, X, y: _r2_score(y, m.predict(X)),
}

def _precision_score(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp) if (tp + fp) > 0 else 0

def _recall_score(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn) if (tp + fn) > 0 else 0

def _f1_score(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

def _r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0

def cross_val_score(estimator, X, y, cv=5, scoring=None, n_jobs=None):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if X.size == 0 or y.size == 0:
        raise ValueError("Empty input")
    if len(X) != len(y):
        raise ValueError("Mismatch in number of samples")
    if isinstance(cv, int):
        if cv < 2:
            raise ValueError("cv must be >= 2")
        unique_y = np.unique(y)
        if len(unique_y) <= 2 and len(unique_y) > 0:
            cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        else:
            cv_obj = KFold(n_splits=cv)
    elif hasattr(cv, 'split'):
        cv_obj = cv
    else:
        raise ValueError("cv must be an int or a CV splitter")
    scores = []
    for train_idx, test_idx in cv_obj.split(X, y):
        model = estimator.__class__()
        init_args = {}
        for k in dir(estimator):
            if not k.startswith('_') and not k.endswith('_') and k not in ('fit', 'predict', 'score', 'predict_proba', 'named_steps'):
                val = getattr(estimator, k)
                if not callable(val):
                    init_args[k] = val
        model.__init__(**init_args)
        model.fit(X[train_idx], y[train_idx])
        if scoring is not None:
            if callable(scoring):
                score = scoring(model, X[test_idx], y[test_idx])
            elif scoring in _SCORERS:
                score = _SCORERS[scoring](model, X[test_idx], y[test_idx])
            else:
                raise ValueError(f"Unknown scoring metric: {scoring}")
        elif hasattr(model, 'score'):
            score = model.score(X[test_idx], y[test_idx])
        else:
            preds = model.predict(X[test_idx])
            score = np.mean(preds == y[test_idx])
        scores.append(score)
    return np.array(scores)

def _fit_and_score_standalone(estimator_class, base_params, params, X, y, train_idx, test_idx):
    if hasattr(estimator_class, 'steps') or 'steps' in base_params:
        steps = base_params.get('steps', [])
        model = estimator_class(steps=steps)
        if params:
            model.set_params(**params)
    else:
        init_args = base_params.copy()
        init_args.update(params)
        model = estimator_class(**init_args)
    model.fit(X[train_idx], y[train_idx])
    if hasattr(model, 'score'):
        score = model.score(X[test_idx], y[test_idx])
    else:
        preds = model.predict(X[test_idx])
        score = np.mean(preds == y[test_idx])
    return score

class GridSearchCV:
    def __init__(self, estimator, param_grid, cv=5, n_jobs=None, scoring=None, refit=True):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.n_jobs = n_jobs if n_jobs is not None else 1
        self.scoring = scoring
        self.refit = refit
        self.best_estimator_ = None
        self.best_params_ = None
        self.best_score_ = -np.inf
        self.cv_results_ = {}

    @_catch_panic
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        if len(X) != len(y):
            raise ValueError("Mismatch in number of samples")

        if isinstance(self.cv, int):
            if self.cv < 2:
                raise ValueError("cv must be >= 2")
            if self.cv > len(X):
                raise ValueError("cv must be <= number of samples")
            cv_obj = KFold(n_splits=self.cv)
        else:
            cv_obj = self.cv

        folds = list(cv_obj.split(X, y))
        
        if not self.param_grid:
            self.best_params_ = {}
            self.best_estimator_ = self.estimator
            self.best_score_ = 0.0
            self.cv_results_ = {"mean_test_score": [0.0], "params": [{}]}
            return self

        keys, values = zip(*self.param_grid.items())
        experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

        best_avg_score = -np.inf
        best_params = None
        
        estimator_class = self.estimator.__class__
        base_params = {}
        for k in dir(self.estimator):
            if not k.startswith('_') and not k.endswith('_') and k not in ('fit', 'predict', 'score', 'predict_proba'):
                val = getattr(self.estimator, k)
                if not callable(val):
                    base_params[k] = val

        mean_scores = []
        if self.n_jobs == 1:
            for params in experiments:
                scores = []
                for train_idx, test_idx in folds:
                    score = _fit_and_score_standalone(estimator_class, base_params, params, X, y, train_idx, test_idx)
                    scores.append(score)
                avg_score = np.mean(scores)
                mean_scores.append(avg_score)
                if avg_score > best_avg_score:
                    best_avg_score = avg_score
                    best_params = params
        else:
            tasks = []
            for p_idx, params in enumerate(experiments):
                for f_idx, (train_idx, test_idx) in enumerate(folds):
                    tasks.append((p_idx, params, train_idx, test_idx))
            
            with ThreadPoolExecutor(max_workers=self.n_jobs if self.n_jobs > 0 else None) as executor:
                futures = [executor.submit(_fit_and_score_standalone, estimator_class, base_params, t[1], X, y, t[2], t[3]) for t in tasks]
                results = [f.result() for f in futures]
            
            scores_by_param = {i: [] for i in range(len(experiments))}
            for i, res in enumerate(results):
                p_idx = tasks[i][0]
                scores_by_param[p_idx].append(res)
            
            for p_idx, scores in scores_by_param.items():
                avg_score = np.mean(scores)
                mean_scores.append(avg_score)
                if avg_score > best_avg_score:
                    best_avg_score = avg_score
                    best_params = experiments[p_idx]

        self.best_params_ = best_params
        self.best_score_ = best_avg_score
        self.cv_results_ = {"mean_test_score": mean_scores, "params": experiments}

        if self.refit:
            if hasattr(self.estimator, 'steps'):
                self.best_estimator_ = self.estimator.__class__(steps=self.estimator.steps)
                if best_params:
                    self.best_estimator_.set_params(**best_params)
            else:
                self.best_estimator_ = self.estimator.__class__()
                init_args = base_params.copy()
                if best_params:
                    init_args.update(best_params)
                self.best_estimator_.__init__(**init_args)
            self.best_estimator_.fit(X, y)
        return self

    @_catch_panic
    def predict(self, X):
        if self.best_estimator_ is not None:
            return self.best_estimator_.predict(X)
        return self.estimator.predict(X)

    def score(self, X, y):
        if self.best_estimator_ is not None:
            return self.best_estimator_.score(X, y)
        return self.estimator.score(X, y)

class SuccessiveHalvingSearchCV:
    def __init__(self, estimator, param_grid, min_resources=10, factor=3):
        self.estimator = estimator
        self.param_grid = param_grid
        self.min_resources = min_resources
        self.factor = factor
        
        # param_grid is a dict or list of dicts. If it's a dict, convert to list of dicts for combinations
        if isinstance(param_grid, dict):
            keys, values = zip(*param_grid.items())
            self._param_list = [dict(zip(keys, v)) for v in itertools.product(*values)]
        else:
            self._param_list = param_grid
            
        self._model = _core.SuccessiveHalvingSearchCV(
            self.estimator,
            self._param_list,
            self.min_resources,
            self.factor
        )

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self._model.fit(X, y)
        return self

    @property
    def best_estimator_(self):
        return self._model.best_estimator_

    @property
    def best_score_(self):
        return self._model.best_score_
