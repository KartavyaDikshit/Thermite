import numpy as np
from . import _core

def accuracy_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return _core.accuracy_score(y_true, y_pred)

def precision_score(y_true, y_pred, *, average="binary"):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return _core.precision_score(y_true, y_pred, average=average)

def recall_score(y_true, y_pred, *, average="binary"):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return _core.recall_score(y_true, y_pred, average=average)

def f1_score(y_true, y_pred, *, average="binary"):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return _core.f1_score(y_true, y_pred, average=average)

def roc_auc_score(y_true, y_score):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    return _core.roc_auc_score(y_true, y_score)

def mean_squared_error(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return _core.mean_squared_error(y_true, y_pred)

def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return _core.r2_score(y_true, y_pred)
