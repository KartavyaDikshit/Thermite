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

class Pipeline:
    def __init__(self, steps):
        self.steps = steps

    @_catch_panic
    def fit(self, X, y=None, **fit_params):
        Xt = X
        for name, transform in self.steps[:-1]:
            if hasattr(transform, "fit_transform"):
                Xt = transform.fit_transform(Xt, y, **fit_params)
            else:
                Xt = transform.fit(Xt, y, **fit_params).transform(Xt)
        self.steps[-1][1].fit(Xt, y, **fit_params)
        return self

    @_catch_panic
    def predict(self, X):
        Xt = X
        for name, transform in self.steps[:-1]:
            Xt = transform.transform(Xt)
        return self.steps[-1][1].predict(Xt)

    @_catch_panic
    def fit_predict(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.predict(X)

    def fit_transform(self, X, y=None, **fit_params):
        Xt = X
        for name, transform in self.steps[:-1]:
            if hasattr(transform, "fit_transform"):
                Xt = transform.fit_transform(Xt, y, **fit_params)
            else:
                Xt = transform.fit(Xt, y, **fit_params).transform(Xt)
        last_step = self.steps[-1][1]
        if hasattr(last_step, "fit_transform"):
            return last_step.fit_transform(Xt, y, **fit_params)
        else:
            return last_step.fit(Xt, y, **fit_params).transform(Xt)

    @_catch_panic
    def predict_proba(self, X):
        Xt = X
        for name, transform in self.steps[:-1]:
            Xt = transform.transform(Xt)
        return self.steps[-1][1].predict_proba(Xt)

    def score(self, X, y):
        Xt = X
        for name, transform in self.steps[:-1]:
            Xt = transform.transform(Xt)
        return self.steps[-1][1].score(Xt, y)

    @_catch_panic
    def transform(self, X):
        Xt = X
        for name, transform in self.steps:
            Xt = transform.transform(Xt)
        return Xt
        
    @property
    def named_steps(self):
        return dict(self.steps)

import numpy as np

class ColumnTransformer:
    def __init__(self, transformers, remainder='drop'):
        self.transformers = transformers
        self.remainder = remainder

    def fit_transform(self, X, y=None):
        X = np.asarray(X)
        results = []
        transformed_cols = set()
        
        for name, transformer, cols in self.transformers:
            X_cols = X[:, cols]
            if hasattr(transformer, 'fit_transform'):
                res = transformer.fit_transform(X_cols, y)
            else:
                res = transformer.fit(X_cols, y).transform(X_cols)
            
            if res.ndim == 1:
                res = res.reshape(-1, 1)
            results.append(res)
            
            if isinstance(cols, int):
                transformed_cols.add(cols)
            elif isinstance(cols, slice):
                # handle slices
                indices = list(range(X.shape[1]))[cols]
                transformed_cols.update(indices)
            else:
                transformed_cols.update(cols)
                
        if self.remainder == 'passthrough':
            all_cols = set(range(X.shape[1]))
            rem_cols = list(all_cols - transformed_cols)
            if rem_cols:
                rem_cols.sort()
                results.append(X[:, rem_cols])
                
        return np.hstack(results) if results else np.empty((X.shape[0], 0))
        
    @_catch_panic
    def transform(self, X):
        X = np.asarray(X)
        results = []
        transformed_cols = set()
        
        for name, transformer, cols in self.transformers:
            X_cols = X[:, cols]
            res = transformer.transform(X_cols)
            if res.ndim == 1:
                res = res.reshape(-1, 1)
            results.append(res)
            
            if isinstance(cols, int):
                transformed_cols.add(cols)
            elif isinstance(cols, slice):
                indices = list(range(X.shape[1]))[cols]
                transformed_cols.update(indices)
            else:
                transformed_cols.update(cols)
                
        if self.remainder == 'passthrough':
            all_cols = set(range(X.shape[1]))
            rem_cols = list(all_cols - transformed_cols)
            if rem_cols:
                rem_cols.sort()
                results.append(X[:, rem_cols])
                
        return np.hstack(results) if results else np.empty((X.shape[0], 0))
        
    @_catch_panic
    def fit(self, X, y=None):
        self.fit_transform(X, y)
        return self
