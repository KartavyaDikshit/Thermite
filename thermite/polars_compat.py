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

"""
thermite.polars_compat - Zero-copy Polars DataFrame support.

Converts Polars DataFrames and Series to numpy arrays using the
Apache Arrow memory format (zero-copy where the dtype allows it),
then passes them to Thermite models transparently.

Usage:
    from thermite.polars_compat import from_polars

    clf.fit(*from_polars(df_train, target_col="label"))
    preds = clf.predict(from_polars_X(df_test))
"""

import numpy as np

try:
    import polars as pl
    _POLARS_AVAILABLE = True
except ImportError:
    _POLARS_AVAILABLE = False


def _require_polars():
    if not _POLARS_AVAILABLE:
        raise ImportError(
            "polars is required for this feature. Install it with: pip install polars"
        )


def from_polars_X(df) -> np.ndarray:
    """
    Convert a Polars DataFrame to a float64 numpy array (zero-copy where possible).

    Parameters
    ----------
    df : polars.DataFrame
        Input feature matrix. All columns must be numeric.

    Returns
    -------
    np.ndarray of shape (n_samples, n_features), dtype=float64
    """
    _require_polars()
    if not isinstance(df, pl.DataFrame):
        raise TypeError(f"Expected polars.DataFrame, got {type(df)}")

    # to_numpy() uses Arrow zero-copy for compatible dtypes
    arr = df.to_numpy()
    return np.ascontiguousarray(arr, dtype=np.float64)


def from_polars_y(series) -> np.ndarray:
    """
    Convert a Polars Series to a float64 numpy array (zero-copy where possible).

    Parameters
    ----------
    series : polars.Series
        Target labels or values.

    Returns
    -------
    np.ndarray of shape (n_samples,), dtype=float64
    """
    _require_polars()
    if not isinstance(series, pl.Series):
        raise TypeError(f"Expected polars.Series, got {type(series)}")

    return np.ascontiguousarray(series.to_numpy(), dtype=np.float64)


def from_polars(df, target_col: str):
    """
    Split a Polars DataFrame into (X, y) numpy arrays for model training.

    Parameters
    ----------
    df : polars.DataFrame
        Full dataset including features and target column.
    target_col : str
        Name of the target column.

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_features), float64
    y : np.ndarray of shape (n_samples,), float64
    """
    _require_polars()
    if not isinstance(df, pl.DataFrame):
        raise TypeError(f"Expected polars.DataFrame, got {type(df)}")

    feature_cols = [c for c in df.columns if c != target_col]
    X = from_polars_X(df.select(feature_cols))
    y = from_polars_y(df[target_col])
    return X, y


def make_polars_pipeline(model):
    """
    Wrap a Thermite model so its fit/predict methods accept Polars DataFrames
    directly in addition to numpy arrays.

    Parameters
    ----------
    model : any Thermite estimator

    Returns
    -------
    PolarsCompatModel wrapping the estimator
    """
    _require_polars()
    return _PolarsCompatModel(model)


class _PolarsCompatModel:
    """
    Thin adapter that transparently converts Polars inputs before
    forwarding to the underlying Thermite estimator.
    """

    def __init__(self, model):
        self._model = model

    def _coerce(self, X):
        if _POLARS_AVAILABLE and isinstance(X, pl.DataFrame):
            return from_polars_X(X)
        return X

    def _coerce_y(self, y):
        if _POLARS_AVAILABLE and isinstance(y, pl.Series):
            return from_polars_y(y)
        return y

    @_catch_panic
    def fit(self, X, y=None, **kwargs):
        self._model.fit(self._coerce(X), self._coerce_y(y) if y is not None else y, **kwargs)
        return self

    @_catch_panic
    def predict(self, X, **kwargs):
        return self._model.predict(self._coerce(X), **kwargs)

    @_catch_panic
    def predict_proba(self, X, **kwargs):
        return self._model.predict_proba(self._coerce(X), **kwargs)

    @_catch_panic
    def transform(self, X, **kwargs):
        return self._model.transform(self._coerce(X), **kwargs)

    def fit_transform(self, X, y=None, **kwargs):
        return self._model.fit_transform(self._coerce(X), self._coerce_y(y) if y is not None else y, **kwargs)

    def score(self, X, y, **kwargs):
        return self._model.score(self._coerce(X), self._coerce_y(y), **kwargs)

    def __getattr__(self, name):
        # Forward any attribute not explicitly defined to the underlying model
        return getattr(self._model, name)

    def __repr__(self):
        return f"PolarsCompat({self._model!r})"
