import numpy as np
from . import _core

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
from concurrent.futures import ProcessPoolExecutor

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

class GridSearchCV:
    def __init__(self, estimator, param_grid, cv=5, n_jobs=None):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.n_jobs = n_jobs if n_jobs is not None else 1
        self.best_estimator_ = None
        self.best_params_ = None
        self.best_score_ = -np.inf

def _fit_and_score_standalone(estimator_class, base_params, params, X, y, train_idx, test_idx):
    # Combine base params with current params
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
    def __init__(self, estimator, param_grid, cv=5, n_jobs=None):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.n_jobs = n_jobs if n_jobs is not None else 1
        self.best_estimator_ = None
        self.best_params_ = None
        self.best_score_ = -np.inf

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        keys, values = zip(*self.param_grid.items())
        experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

        if isinstance(self.cv, int):
            cv_obj = KFold(n_splits=self.cv)
        else:
            cv_obj = self.cv

        folds = list(cv_obj.split(X, y))
        
        best_avg_score = -np.inf
        best_params = None
        
        estimator_class = self.estimator.__class__
        # get the original non-grid params from the estimator to rebuild it
        base_params = {}
        for k in dir(self.estimator):
            if not k.startswith('_') and not k.endswith('_') and k not in ('fit', 'predict', 'score', 'predict_proba'):
                base_params[k] = getattr(self.estimator, k)

        if self.n_jobs == 1:
            for params in experiments:
                scores = []
                for train_idx, test_idx in folds:
                    score = _fit_and_score_standalone(estimator_class, base_params, params, X, y, train_idx, test_idx)
                    scores.append(score)
                avg_score = np.mean(scores)
                if avg_score > best_avg_score:
                    best_avg_score = avg_score
                    best_params = params
        else:
            # Parallel execution
            tasks = []
            for p_idx, params in enumerate(experiments):
                for f_idx, (train_idx, test_idx) in enumerate(folds):
                    tasks.append((p_idx, params, train_idx, test_idx))
            
            with ProcessPoolExecutor(max_workers=self.n_jobs if self.n_jobs > 0 else None) as executor:
                futures = [executor.submit(_fit_and_score_standalone, estimator_class, base_params, t[1], X, y, t[2], t[3]) for t in tasks]
                results = [f.result() for f in futures]
            
            # Aggregate results
            scores_by_param = {i: [] for i in range(len(experiments))}
            for i, res in enumerate(results):
                p_idx = tasks[i][0]
                scores_by_param[p_idx].append(res)
            
            for p_idx, scores in scores_by_param.items():
                avg_score = np.mean(scores)
                if avg_score > best_avg_score:
                    best_avg_score = avg_score
                    best_params = experiments[p_idx]

        self.best_params_ = best_params
        self.best_score_ = best_avg_score

        # Refit on full dataset
        self.best_estimator_ = self.estimator.__class__()
        init_args = {}
        if hasattr(self.best_estimator_, "_model"):
            import inspect
            sig = inspect.signature(self.best_estimator_.__class__.__init__)
            for k in sig.parameters:
                if k != 'self':
                    init_args[k] = self.best_params_.get(k, getattr(self.estimator, k, None))
        self.best_estimator_.__init__(**init_args)
        
        self.best_estimator_.fit(X, y)
        return self

    def predict(self, X):
        return self.best_estimator_.predict(X)
