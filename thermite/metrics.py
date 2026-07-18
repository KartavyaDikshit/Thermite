import numpy as np
import warnings
from . import _core

def _validate_inputs(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0 or y_pred.size == 0:
        raise ValueError("Empty input")
    if y_true.shape != y_pred.shape:
        raise ValueError("Shape mismatch")
    return y_true, y_pred

def _to_float(y):
    if np.issubdtype(y.dtype, np.str_) or np.issubdtype(y.dtype, np.bytes_):
        return np.array([float(ord(s[0]) if isinstance(s, str) and len(s) == 1 else hash(s) % 1000) for s in y], dtype=np.float64)
    return np.asarray(y, dtype=np.float64)

def accuracy_score(y_true, y_pred, *, normalize=True, sample_weight=None):
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    y_true_f = _to_float(y_true)
    y_pred_f = _to_float(y_pred)
    result = _core.accuracy_score(y_true_f, y_pred_f)
    if not normalize:
        result = int(result * len(y_true))
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
        correct = np.isclose(y_true_f.ravel(), y_pred_f.ravel())
        result = np.average(correct, weights=sample_weight)
    return result

def _precision_recall_f1_binary(y_true, y_pred, pos_label, metric_fn, average, zero_division):
    y_true_a = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred)
    if np.issubdtype(y_true_a.dtype, np.str_) or np.issubdtype(y_true_a.dtype, np.bytes_) or isinstance(pos_label, str):
        pos = pos_label if isinstance(pos_label, str) else str(pos_label)
        y_true_bin = np.array([1.0 if str(v) == pos else 0.0 for v in y_true_a.ravel()], dtype=np.float64)
        y_pred_bin = np.array([1.0 if str(v) == pos else 0.0 for v in y_pred_a.ravel()], dtype=np.float64)
    else:
        y_true_f = np.asarray(y_true, dtype=np.float64)
        y_pred_f = np.asarray(y_pred, dtype=np.float64)
        if float(pos_label) != 1.0:
            y_true_bin = np.where(np.isclose(y_true_f, float(pos_label)), 1.0, 0.0)
            y_pred_bin = np.where(np.isclose(y_pred_f, float(pos_label)), 1.0, 0.0)
        else:
            y_true_bin = y_true_f
            y_pred_bin = y_pred_f
    try:
        return metric_fn(y_true_bin, y_pred_bin, average=average)
    except BaseException:
        if zero_division == "warn" or zero_division == 0.0:
            return 0.0
        return 1.0

def precision_score(y_true, y_pred, *, average="binary", pos_label=1, sample_weight=None, zero_division="warn"):
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    if yt.size == 0:
        raise ValueError("Empty input")
    if yt.shape != yp.shape:
        raise ValueError("Shape mismatch")
    return _precision_recall_f1_binary(yt, yp, pos_label, _core.precision_score, average, zero_division)

def recall_score(y_true, y_pred, *, average="binary", pos_label=1, sample_weight=None, zero_division="warn"):
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    if yt.size == 0:
        raise ValueError("Empty input")
    if yt.shape != yp.shape:
        raise ValueError("Shape mismatch")
    return _precision_recall_f1_binary(yt, yp, pos_label, _core.recall_score, average, zero_division)

def f1_score(y_true, y_pred, *, average="binary", pos_label=1, sample_weight=None, zero_division="warn"):
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    if yt.size == 0:
        raise ValueError("Empty input")
    if yt.shape != yp.shape:
        raise ValueError("Shape mismatch")
    return _precision_recall_f1_binary(yt, yp, pos_label, _core.f1_score, average, zero_division)

def roc_auc_score(y_true, y_score, *, average="macro", sample_weight=None, multi_class="raise"):
    y_true = np.ascontiguousarray(np.asarray(y_true, dtype=np.float64))
    y_score = np.ascontiguousarray(np.asarray(y_score, dtype=np.float64))
    if y_true.size == 0:
        raise ValueError("Empty input")
    if len(np.unique(y_true)) < 2:
        raise ValueError("ROC AUC requires at least 2 classes")
    return _core.roc_auc_score(y_true, y_score)

def mean_squared_error(y_true, y_pred, *, sample_weight=None, multioutput="uniform_average"):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size == 0:
        raise ValueError("Empty input")
    if y_true.shape != y_pred.shape:
        raise ValueError("Shape mismatch")
    if y_true.ndim == 2:
        result = np.zeros(y_true.shape[1], dtype=np.float64)
        for i in range(y_true.shape[1]):
            result[i] = _core.mean_squared_error(np.ascontiguousarray(y_true[:, i]), np.ascontiguousarray(y_pred[:, i]))
        if multioutput == "raw_values":
            return result
        return float(result.mean())
    return _core.mean_squared_error(y_true, y_pred)

def r2_score(y_true, y_pred, *, sample_weight=None, multioutput="uniform_average"):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size == 0:
        raise ValueError("Empty input")
    if y_true.shape != y_pred.shape:
        raise ValueError("Shape mismatch")
    if y_true.ndim == 1 and len(y_true) == 1:
        import warnings as _w
        _w.warn("R2 score with a single sample")
        return np.float64(np.nan)
    if np.var(y_true) < 1e-15:
        if np.allclose(y_true, y_pred):
            return 1.0
        return 0.0
    if y_true.ndim == 2:
        result = np.zeros(y_true.shape[1], dtype=np.float64)
        for i in range(y_true.shape[1]):
            result[i] = _core.r2_score(np.ascontiguousarray(y_true[:, i]), np.ascontiguousarray(y_pred[:, i]))
        if multioutput == "raw_values":
            return result
        return float(result.mean())
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