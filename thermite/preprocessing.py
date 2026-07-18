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


class StandardScaler:
    def __init__(self, *, with_mean=True, with_std=True):
        self._scaler = _core.StandardScaler(with_mean=with_mean, with_std=with_std)
        
    @_catch_panic
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        self._scaler.fit(X)
        return self
        
    @_catch_panic
    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        return self._scaler.transform(X)
        
    def fit_transform(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        return self._scaler.fit_transform(X)
        
    def inverse_transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        return self._scaler.inverse_transform(X)
        
    @property
    def mean_(self):
        return self._scaler.mean
        
    @property
    def var_(self):
        return self._scaler.var
        
    @property
    def scale_(self):
        return self._scaler.scale
        
    @property
    def n_samples_seen_(self):
        return self._scaler.n_samples_seen

class MinMaxScaler:
    def __init__(self, feature_range=(0.0, 1.0)):
        self._scaler = _core.MinMaxScaler(feature_range=feature_range)
        
    @_catch_panic
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        self._scaler.fit(X)
        return self
        
    @_catch_panic
    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        return self._scaler.transform(X)
        
    def fit_transform(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        return self._scaler.fit_transform(X)
        
    def inverse_transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        return self._scaler.inverse_transform(X)
        
    @property
    def data_min_(self):
        return self._scaler.data_min
        
    @property
    def data_max_(self):
        return self._scaler.data_max
        
    @property
    def feature_range(self):
        return self._scaler.feature_range

    @feature_range.setter
    def feature_range(self, value):
        self._scaler.feature_range = value

    @property
    def scale_(self):
        return self._scaler.scale
        
    @property
    def min_(self):
        return self._scaler.min

class LabelEncoder:
    def __init__(self):
        self._encoder = _core.LabelEncoder()
        self.classes_ = None
        
    def fit(self, y):
        y = np.asarray(y)
        if y.size == 0:
            self.classes_ = np.array([])
            return self
        if y.ndim != 1:
            raise ValueError("Expected 1D array for y")
            
        if np.issubdtype(y.dtype, np.integer):
            y_cast = y.astype(np.int64)
            self._encoder.fit_int(y_cast)
            self.classes_ = np.array(self._encoder.get_classes_int())
        elif np.issubdtype(y.dtype, np.floating):
            y_cast = y.astype(np.float64)
            self._encoder.fit_float(y_cast)
            self.classes_ = np.array(self._encoder.get_classes_float())
        else:
            y_cast = list(y.astype(str))
            self._encoder.fit_str(y_cast)
            self.classes_ = np.array(self._encoder.get_classes_str())
        return self
        
    def transform(self, y):
        y = np.asarray(y)
        if y.size == 0:
            return np.array([])
        if y.ndim != 1:
            raise ValueError("Expected 1D array for y")
            
        if np.issubdtype(y.dtype, np.integer):
            y_cast = y.astype(np.int64)
            return self._encoder.transform_int(y_cast)
        elif np.issubdtype(y.dtype, np.floating):
            y_cast = y.astype(np.float64)
            return self._encoder.transform_float(y_cast)
        else:
            y_cast = list(y.astype(str))
            return self._encoder.transform_str(y_cast)
            
    def fit_transform(self, y):
        return self.fit(y).transform(y)
        
    def inverse_transform(self, y):
        y = np.asarray(y, dtype=np.int64)
        if y.ndim != 1:
            raise ValueError("Expected 1D array for y")
            
        if self.classes_.dtype.kind in ('i', 'u'):
            return self._encoder.inverse_transform_int(y)
        elif self.classes_.dtype.kind in ('f',):
            return self._encoder.inverse_transform_float(y)
        else:
            return np.array(self._encoder.inverse_transform_str(y))

class OneHotEncoder:
    def __init__(self, *, categories="auto", drop=None, sparse_output=True, handle_unknown="error", **kwargs):
        if handle_unknown not in ("error", "ignore"):
            raise ValueError("handle_unknown must be 'error' or 'ignore'")
        self._categories_param = categories
        self.drop = drop
        self.sparse_output = sparse_output
        self.handle_unknown = handle_unknown
        self._encoder = _core.OneHotEncoder(handle_unknown=handle_unknown) if categories == "auto" else None
        self.categories_ = None
        
    def _one_hot_encode(self, X, categories):
        n_samples = X.shape[0]
        n_cats_per_feature = [len(c) for c in categories]
        total_cols = sum(n_cats_per_feature)
        result = np.zeros((n_samples, total_cols), dtype=np.float64)
        col_offset = 0
        for feat_idx in range(X.shape[1]):
            n_cats = len(categories[feat_idx])
            for cat_idx, cat_val in enumerate(categories[feat_idx]):
                for row_idx in range(n_samples):
                    if str(X[row_idx, feat_idx]) == str(cat_val):
                        result[row_idx, col_offset + cat_idx] = 1.0
            col_offset += n_cats
        return result

    def fit(self, X, y=None):
        X = np.asarray(X)
        if X.size == 0:
            raise ValueError("Empty input")
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
            
        if self._categories_param == "auto":
            if np.issubdtype(X.dtype, np.integer):
                self._encoder.fit_int(X.astype(np.int64))
            else:
                X_str = X.astype(str)
                cols = [list(X_str[:, i]) for i in range(X_str.shape[1])]
                self._encoder.fit_str(cols)
            raw_cats = self._encoder.categories
            self.categories_ = [np.array(c) for c in raw_cats]
        else:
            self.categories_ = [np.array(c) for c in self._categories_param]

        return self
        
    def transform(self, X):
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")

        if self._categories_param == "auto":
            if np.issubdtype(X.dtype, np.integer):
                result = self._encoder.transform_int(X.astype(np.int64))
            else:
                X_str = X.astype(str)
                cols = [list(X_str[:, i]) for i in range(X_str.shape[1])]
                result = self._encoder.transform_str(cols)
        else:
            result = self._one_hot_encode(X, self.categories_)

        if self.drop == 'first':
            n_cats_per_feature = [len(c) for c in self.categories_]
            cols_to_drop = []
            offset = 0
            for n in n_cats_per_feature:
                cols_to_drop.append(offset)
                offset += n
            result = np.delete(result, cols_to_drop, axis=1)

        return result
            
    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
        
    def inverse_transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
            
        if self.categories_ is None or len(self.categories_) == 0:
            raise ValueError("OneHotEncoder is not fitted yet")
            
        is_int = np.issubdtype(self.categories_[0].dtype, np.integer)
        if is_int:
            return self._encoder.inverse_transform_int(X)
        else:
            return np.array(self._encoder.inverse_transform_str(X))
