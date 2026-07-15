import numpy as np
import warnings
from . import _core

def _validate_inputs(y_true, y_pred, multioutput=False):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.ndim == 0 or y_pred.ndim == 0:
        raise ValueError("Input must be 1D or 2D arrays")
    if y_true.size == 0:
        raise ValueError("Empty input")
    return y_true, y_pred

def accuracy_score(y_true, y_pred, *, normalize=True, sample_weight=None):
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
    result = _core.accuracy_score(y_true, y_pred)
    if not normalize:
        result = result * len(y_true)
    if sample_weight is not None:
        result = np.average([y_true[i] == y_pred[i] for i in range(len(y_true))], weights=sample_weight)
    return result

def precision_score(y_true, y_pred, *, average="binary", pos_label=1, sample_weight=None, zero_division="warn"):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
    if zero_division == "warn":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                return _core.precision_score(y_true, y_pred, average=average)
            except BaseException:
                return 0.0
    return _core.precision_score(y_true, y_pred, average=average)

def recall_score(y_true, y_pred, *, average="binary", pos_label=1, sample_weight=None, zero_division="warn"):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
    if zero_division == "warn":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                return _core.recall_score(y_true, y_pred, average=average)
            except BaseException:
                return 0.0
    return _core.recall_score(y_true, y_pred, average=average)

def f1_score(y_true, y_pred, *, average="binary", pos_label=1, sample_weight=None, zero_division="warn"):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
    if zero_division == "warn":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                return _core.f1_score(y_true, y_pred, average=average)
            except BaseException:
                return 0.0
    return _core.f1_score(y_true, y_pred, average=average)

def roc_auc_score(y_true, y_score, *, average="macro", sample_weight=None, multi_class="raise"):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
    return _core.roc_auc_score(y_true, y_score)

def mean_squared_error(y_true, y_pred, *, sample_weight=None, multioutput="uniform_average"):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
    if multioutput == "raw_values":
        return _core.mean_squared_error(y_true, y_pred)
    return _core.mean_squared_error(y_true, y_pred)

def r2_score(y_true, y_pred, *, sample_weight=None, multioutput="uniform_average"):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
    if multioutput == "raw_values":
        return _core.r2_score(y_true, y_pred)
    return _core.r2_score(y_true, y_pred)

def log_loss(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return _core.log_loss(y_true, y_pred)

def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return _core.mean_absolute_percentage_error(y_true, y_pred)

def pairwise_distances(X, Y, metric="cosine"):
    X = np.ascontiguousarray(X, dtype=np.float64)
    Y = np.ascontiguousarray(Y, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if Y.ndim == 1:
        Y = Y.reshape(1, -1)
    return _core.pairwise_distances(X, Y, metric)