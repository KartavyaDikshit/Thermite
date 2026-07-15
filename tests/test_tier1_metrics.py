import pytest
import numpy as np
from tests.conftest import get_module

# Dynamically import metrics module
metrics = get_module("metrics")


# accuracy_score tests
def test_accuracy_basic():
    """Binary classification accuracy."""
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 0]
    score = metrics.accuracy_score(y_true, y_pred)
    np.testing.assert_allclose(score, 0.75)


def test_accuracy_multiclass():
    """Multi-class classification accuracy."""
    y_true = [0, 1, 2, 2]
    y_pred = [0, 2, 2, 1]
    score = metrics.accuracy_score(y_true, y_pred)
    np.testing.assert_allclose(score, 0.5)


def test_accuracy_normalize():
    """Check when normalize=False (returns count of correct predictions)."""
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 0]
    score = metrics.accuracy_score(y_true, y_pred, normalize=False)
    assert score == 3


def test_accuracy_sample_weight():
    """Check with custom sample weights."""
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 0]
    # Weight the mistake (index 2) to 0.0, making the weighted accuracy 1.0
    score = metrics.accuracy_score(y_true, y_pred, sample_weight=[1.0, 1.0, 0.0, 1.0])
    np.testing.assert_allclose(score, 1.0)


def test_accuracy_perfect():
    """Check accuracy is exactly 1.0 when predictions match targets exactly."""
    y_true = np.array([2, 1, 0, 5])
    score = metrics.accuracy_score(y_true, y_true)
    np.testing.assert_allclose(score, 1.0)


# precision_score tests
def test_precision_basic():
    """Binary precision."""
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 1, 1]  # 2 TP, 1 FP -> prec = 2/3
    score = metrics.precision_score(y_true, y_pred)
    np.testing.assert_allclose(score, 2/3)


def test_precision_average():
    """Check precision under different multiclass averaging options."""
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 2, 1, 0, 0, 1]
    
    for avg in ["macro", "micro", "weighted"]:
        score = metrics.precision_score(y_true, y_pred, average=avg)
        assert 0.0 <= score <= 1.0


def test_precision_zero_division():
    """Check zero_division parameter."""
    y_true = [0, 0, 0]
    y_pred = [0, 0, 0]
    
    score_zero = metrics.precision_score(y_true, y_pred, zero_division=0.0)
    score_one = metrics.precision_score(y_true, y_pred, zero_division=1.0)
    
    np.testing.assert_allclose(score_zero, 0.0)
    np.testing.assert_allclose(score_one, 1.0)


def test_precision_pos_label():
    """Check changing pos_label."""
    y_true = ["a", "b", "a", "b"]
    y_pred = ["a", "b", "b", "b"]
    
    # If pos_label is "a": TP=1, FP=0, FN=1 -> prec = 1/1 = 1.0
    score_a = metrics.precision_score(y_true, y_pred, pos_label="a")
    # If pos_label is "b": TP=2, FP=1, FN=0 -> prec = 2/3
    score_b = metrics.precision_score(y_true, y_pred, pos_label="b")
    
    np.testing.assert_allclose(score_a, 1.0)
    np.testing.assert_allclose(score_b, 2/3)


def test_precision_sample_weight():
    """Check precision with sample weights."""
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 1, 1]  # TP at 1, 3. FP at 2.
    # zero weight on the FP (index 2)
    score = metrics.precision_score(y_true, y_pred, sample_weight=[1.0, 1.0, 0.0, 1.0])
    np.testing.assert_allclose(score, 1.0)


# recall_score tests
def test_recall_basic():
    """Binary recall."""
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 0]  # 1 TP, 1 FN -> recall = 0.5
    score = metrics.recall_score(y_true, y_pred)
    np.testing.assert_allclose(score, 0.5)


def test_recall_average():
    """Check recall under different multiclass averaging options."""
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 2, 1, 0, 0, 1]
    
    for avg in ["macro", "micro", "weighted"]:
        score = metrics.recall_score(y_true, y_pred, average=avg)
        assert 0.0 <= score <= 1.0


def test_recall_zero_division():
    """Check zero_division parameter for recall."""
    y_true = [0, 0, 0]
    y_pred = [1, 1, 1]
    # No positive true labels, so recall denominator is 0
    score_zero = metrics.recall_score(y_true, y_pred, zero_division=0.0)
    score_one = metrics.recall_score(y_true, y_pred, zero_division=1.0)
    
    np.testing.assert_allclose(score_zero, 0.0)
    np.testing.assert_allclose(score_one, 1.0)


def test_recall_pos_label():
    """Check recall with custom pos_label."""
    y_true = ["a", "b", "a", "b"]
    y_pred = ["a", "b", "b", "b"]
    
    # If pos_label is "a": TP=1, FP=0, FN=1 -> recall = 0.5
    score_a = metrics.recall_score(y_true, y_pred, pos_label="a")
    # If pos_label is "b": TP=2, FP=1, FN=0 -> recall = 1.0
    score_b = metrics.recall_score(y_true, y_pred, pos_label="b")
    
    np.testing.assert_allclose(score_a, 0.5)
    np.testing.assert_allclose(score_b, 1.0)


def test_recall_sample_weight():
    """Check recall with sample weights."""
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 0]  # TP at 1, FN at 3.
    # zero weight on the FN (index 3)
    score = metrics.recall_score(y_true, y_pred, sample_weight=[1.0, 1.0, 1.0, 0.0])
    np.testing.assert_allclose(score, 1.0)


# f1_score tests
def test_f1_basic():
    """Binary F1 score."""
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 1, 1]  # TP=2, FP=1, FN=0 -> prec=2/3, rec=1.0 -> f1=2*(2/3*1)/(2/3+1) = 4/3 / 5/3 = 0.8
    score = metrics.f1_score(y_true, y_pred)
    np.testing.assert_allclose(score, 0.8)


def test_f1_average():
    """Check F1 score under different multiclass averaging options."""
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 2, 1, 0, 0, 1]
    
    for avg in ["macro", "micro", "weighted"]:
        score = metrics.f1_score(y_true, y_pred, average=avg)
        assert 0.0 <= score <= 1.0


def test_f1_zero_division():
    """Check zero_division parameter for F1."""
    y_true = [0, 0, 0]
    y_pred = [0, 0, 0]
    
    score_zero = metrics.f1_score(y_true, y_pred, zero_division=0.0)
    score_one = metrics.f1_score(y_true, y_pred, zero_division=1.0)
    
    np.testing.assert_allclose(score_zero, 0.0)
    np.testing.assert_allclose(score_one, 1.0)



def test_f1_pos_label():
    """Check F1 with custom pos_label."""
    y_true = ["a", "b", "a", "b"]
    y_pred = ["a", "b", "b", "b"]
    
    score_a = metrics.f1_score(y_true, y_pred, pos_label="a")
    score_b = metrics.f1_score(y_true, y_pred, pos_label="b")
    
    # pos_label="a": prec=1.0, rec=0.5 -> f1 = 2 * 0.5 / 1.5 = 2/3
    np.testing.assert_allclose(score_a, 2/3)
    # pos_label="b": prec=2/3, rec=1.0 -> f1 = 0.8
    np.testing.assert_allclose(score_b, 0.8)


def test_f1_sample_weight():
    """Check F1 with sample weights."""
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 1, 1]
    score = metrics.f1_score(y_true, y_pred, sample_weight=[1.0, 1.0, 0.0, 1.0])
    np.testing.assert_allclose(score, 1.0)


# roc_auc_score tests
def test_roc_auc_basic():
    """Binary ROC AUC score with probability scores."""
    y_true = [0, 0, 1, 1]
    y_scores = [0.1, 0.4, 0.35, 0.8]
    score = metrics.roc_auc_score(y_true, y_scores)
    # y_scores rank: 0.1 (class 0), 0.35 (class 1), 0.4 (class 0), 0.8 (class 1)
    # One inversion (0.35 is class 1 but scored lower than 0.4 which is class 0)
    # ROC AUC should be 0.75
    np.testing.assert_allclose(score, 0.75)


def test_roc_auc_multiclass():
    """Multiclass ROC AUC (ovr or ovo)."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    # Probability matrix of shape (n_samples, n_classes)
    y_scores = np.array([
        [0.8, 0.1, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.2, 0.7],
        [0.6, 0.2, 0.2],
        [0.1, 0.8, 0.1],
        [0.2, 0.1, 0.7]
    ])
    score_ovr = metrics.roc_auc_score(y_true, y_scores, multi_class="ovr", average="macro")
    score_ovo = metrics.roc_auc_score(y_true, y_scores, multi_class="ovo", average="macro")
    assert 0.5 <= score_ovr <= 1.0
    assert 0.5 <= score_ovo <= 1.0


def test_roc_auc_average():
    """Check average options for multiclass ROC AUC."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_scores = np.array([
        [0.8, 0.1, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.2, 0.7],
        [0.6, 0.2, 0.2],
        [0.1, 0.8, 0.1],
        [0.2, 0.1, 0.7]
    ])
    for avg in ["macro", "weighted"]:
        score = metrics.roc_auc_score(y_true, y_scores, multi_class="ovr", average=avg)
        assert 0.0 <= score <= 1.0


def test_roc_auc_sample_weight():
    """Check ROC AUC with sample weights."""
    y_true = [0, 0, 1, 1]
    y_scores = [0.1, 0.4, 0.35, 0.8]
    # Zeroing out the misranked class 1 sample at index 2
    score = metrics.roc_auc_score(y_true, y_scores, sample_weight=[1.0, 1.0, 0.0, 1.0])
    np.testing.assert_allclose(score, 1.0)


def test_roc_auc_perfect():
    """Check perfect classification yields 1.0."""
    y_true = [0, 0, 1, 1]
    y_scores = [0.1, 0.2, 0.8, 0.9]
    score = metrics.roc_auc_score(y_true, y_scores)
    np.testing.assert_allclose(score, 1.0)


# mean_squared_error tests
def test_mse_basic():
    """Basic mean squared error calculation."""
    y_true = [3.0, -0.5, 2.0, 7.0]
    y_pred = [2.5, 0.0, 2.0, 8.0]
    # errors: 0.5, -0.5, 0.0, -1.0
    # squared: 0.25, 0.25, 0.0, 1.0 -> sum = 1.5 -> mean = 0.375
    score = metrics.mean_squared_error(y_true, y_pred)
    np.testing.assert_allclose(score, 0.375)


def test_mse_multioutput():
    """Multi-output MSE options."""
    y_true = [[0.5, 1.0], [-1.0, 1.0], [7.0, -6.0]]
    y_pred = [[0.0, 2.0], [-1.0, 2.0], [8.0, -5.0]]
    
    score_raw = metrics.mean_squared_error(y_true, y_pred, multioutput="raw_values")
    score_avg = metrics.mean_squared_error(y_true, y_pred, multioutput="uniform_average")
    
    assert score_raw.shape == (2,)
    assert isinstance(score_avg, float)


def test_mse_sample_weight():
    """Check MSE sample weights."""
    y_true = [3.0, -0.5]
    y_pred = [2.5, 0.0]  # errors: 0.5 (squared: 0.25), -0.5 (squared: 0.25)
    # Weighted: zero weight on the second sample
    score = metrics.mean_squared_error(y_true, y_pred, sample_weight=[1.0, 0.0])
    np.testing.assert_allclose(score, 0.25)


def test_mse_perfect():
    """MSE should be 0.0 for identical arrays."""
    y_true = np.array([1.5, -2.4, 0.0])
    score = metrics.mean_squared_error(y_true, y_true)
    np.testing.assert_allclose(score, 0.0, atol=1e-7)


def test_mse_types():
    """Verify MSE works with mixed types (ints and floats)."""
    y_true = np.array([1, 2, 3], dtype=np.int32)
    y_pred = np.array([1.5, 2.0, 2.5], dtype=np.float64)
    score = metrics.mean_squared_error(y_true, y_pred)
    # errors: 0.5, 0.0, 0.5 -> squared: 0.25, 0.0, 0.25 -> mean = 0.166666...
    np.testing.assert_allclose(score, 0.5 / 3)


# r2_score tests
def test_r2_basic():
    """Basic R2 score."""
    y_true = [3.0, -0.5, 2.0, 7.0]
    y_pred = [2.5, 0.0, 2.0, 8.0]
    score = metrics.r2_score(y_true, y_pred)
    assert score <= 1.0


def test_r2_multioutput():
    """Multi-output R2 options."""
    y_true = [[0.5, 1.0], [-1.0, 1.0], [7.0, -6.0]]
    y_pred = [[0.0, 2.0], [-1.0, 2.0], [8.0, -5.0]]
    
    score_raw = metrics.r2_score(y_true, y_pred, multioutput="raw_values")
    score_weighted = metrics.r2_score(y_true, y_pred, multioutput="variance_weighted")
    
    assert score_raw.shape == (2,)
    assert isinstance(score_weighted, float)


def test_r2_sample_weight():
    """Check R2 with sample weights."""
    y_true = [3.0, -0.5, 2.0, 7.0]
    y_pred = [2.5, 0.0, 2.0, 8.0]
    score = metrics.r2_score(y_true, y_pred, sample_weight=[1.0, 1.0, 1.0, 1.0])
    assert isinstance(score, float)


def test_r2_perfect():
    """Perfect predictions yield 1.0."""
    y_true = [1.0, 2.0, 3.0]
    score = metrics.r2_score(y_true, y_true)
    np.testing.assert_allclose(score, 1.0)


def test_r2_negative():
    """Bad predictions yield negative/suboptimal R2."""
    y_true = [1.0, 2.0, 3.0]
    # Predict all zeros, worst than predicting the mean (which would be 2.0)
    y_pred = [0.0, 0.0, 0.0]
    score = metrics.r2_score(y_true, y_pred)
    assert score < 0.0
