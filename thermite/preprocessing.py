import numpy as np
from . import _core

class StandardScaler:
    def __init__(self, *, with_mean=True, with_std=True):
        self._scaler = _core.StandardScaler(with_mean=with_mean, with_std=with_std)
        
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        self._scaler.fit(X)
        return self
        
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
        
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        self._scaler.fit(X)
        return self
        
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
    def __init__(self, *, handle_unknown="error"):
        if handle_unknown not in ("error", "ignore"):
            raise ValueError("handle_unknown must be 'error' or 'ignore'")
        self._encoder = _core.OneHotEncoder(handle_unknown=handle_unknown)
        self.categories_ = None
        
    def fit(self, X, y=None):
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
            
        if np.issubdtype(X.dtype, np.integer):
            self._encoder.fit_int(X.astype(np.int64))
        else:
            X_str = X.astype(str)
            cols = [list(X_str[:, i]) for i in range(X_str.shape[1])]
            self._encoder.fit_str(cols)
            
        raw_cats = self._encoder.categories
        self.categories_ = [np.array(c) for c in raw_cats]
        return self
        
    def transform(self, X):
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
            
        if np.issubdtype(X.dtype, np.integer):
            return self._encoder.transform_int(X.astype(np.int64))
        else:
            X_str = X.astype(str)
            cols = [list(X_str[:, i]) for i in range(X_str.shape[1])]
            return self._encoder.transform_str(cols)
            
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
