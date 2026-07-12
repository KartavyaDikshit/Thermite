import numpy as np
import pytest
from tests.conftest import get_module

# Dynamically import metrics module
metrics = get_module("metrics")


# =====================================================================
# accuracy_score Boundary Cases
# =====================================================================

def test_accuracy_score_empty_input():
    """1. Empty input raises ValueError."""
    with pytest.raises(ValueError):
        metrics.accuracy_score([], [])


def test_accuracy_score_mismatch_shapes():
    """2. Mismatching shapes between true and pred raises ValueError."""
    with pytest.raises(ValueError):
        metrics.accuracy_score([1, 2], [1, 2, 3])


def test_accuracy_score_single_sample():
    """3. Single sample input returns 1.0 (if match) or 0.0 (if mismatch)."""
    assert metrics.accuracy_score([5], [5]) == 1.0
    assert metrics.accuracy_score([5], [3]) == 0.0


def test_accuracy_score_normalize_false():
    """4. Running with normalize=False returns integer match count."""
    y_true = [0, 1, 2, 3]
    y_pred = [0, 2, 2, 4]
    count = metrics.accuracy_score(y_true, y_pred, normalize=False)
    assert count == 2


def test_accuracy_score_sample_weight():
    """5. Accuracy with zero/custom sample weights."""
    y_true = [0, 1, 0, 1]
    y_pred = [0, 0, 0, 0]  # mismatch at index 1 and 3
    # Mismatch at index 3 is weighted 0, mismatch at index 1 is weighted 1
    # Matches at index 0 and 2 are weighted 1 and 1
    # Total correct weight = 2 (indices 0, 2)
    # Total weight = 3
    score = metrics.accuracy_score(y_true, y_pred, sample_weight=[1.0, 1.0, 1.0, 0.0])
    np.testing.assert_allclose(score, 2.0 / 3.0)


# =====================================================================
# precision_score Boundary Cases
# =====================================================================

def test_precision_score_empty_input():
    """1. Empty input raises ValueError."""
    with pytest.raises(ValueError):
        metrics.precision_score([], [])


def test_precision_score_mismatch_shapes():
    """2. Mismatching shapes raises ValueError."""
    with pytest.raises(ValueError):
        metrics.precision_score([1, 0], [1, 0, 1])


def test_precision_score_zero_division_default():
    """3. Default zero_division behavior returns 0.0 with warning on no predicted positive samples."""
    with pytest.warns(UserWarning):
        score = metrics.precision_score([0, 1], [0, 0])
    assert score == 0.0


def test_precision_score_zero_division_custom():
    """4. zero_division parameter set to 1.0 or np.nan controls the output value."""
    score_one = metrics.precision_score([0, 1], [0, 0], zero_division=1.0)
    assert score_one == 1.0
    
    score_nan = metrics.precision_score([0, 1], [0, 0], zero_division=np.nan)
    assert np.isnan(score_nan)


def test_precision_score_multiclass_missing_class():
    """5. Multi-class macro-average precision when some classes have no instances."""
    y_true = [0, 1, 2]
    y_pred = [0, 1, 1]  # class 2 has no predictions
    # precision for class 0: 1.0
    # precision for class 1: 0.5
    # precision for class 2: 0.0 (warning, division by zero)
    with pytest.warns(UserWarning):
        score = metrics.precision_score(y_true, y_pred, average="macro")
    np.testing.assert_allclose(score, (1.0 + 0.5 + 0.0) / 3.0)


# =====================================================================
# recall_score Boundary Cases
# =====================================================================

def test_recall_score_empty_input():
    """1. Empty input raises ValueError."""
    with pytest.raises(ValueError):
        metrics.recall_score([], [])


def test_recall_score_mismatch_shapes():
    """2. Mismatching shapes raises ValueError."""
    with pytest.raises(ValueError):
        metrics.recall_score([1, 0], [1, 0, 1])


def test_recall_score_zero_division_default():
    """3. Default zero_division behavior returns 0.0 with warning on no true positive samples."""
    with pytest.warns(UserWarning):
        score = metrics.recall_score([0, 0], [0, 1])
    assert score == 0.0


def test_recall_score_zero_division_custom():
    """4. zero_division parameter set to 1.0 or np.nan controls the output value."""
    score_one = metrics.recall_score([0, 0], [0, 1], zero_division=1.0)
    assert score_one == 1.0
    
    score_nan = metrics.recall_score([0, 0], [0, 1], zero_division=np.nan)
    assert np.isnan(score_nan)


def test_recall_score_multiclass_missing_class():
    """5. Multi-class macro-average recall when some classes have no true instances."""
    y_true = [0, 1, 0]  # class 2 has no true instances
    y_pred = [0, 1, 2]
    # recall for class 0: 0.5
    # recall for class 1: 1.0
    # recall for class 2: 0.0 (warning, division by zero)
    with pytest.warns(UserWarning):
        score = metrics.recall_score(y_true, y_pred, average="macro")
    np.testing.assert_allclose(score, (0.5 + 1.0 + 0.0) / 3.0)


# =====================================================================
# f1_score Boundary Cases
# =====================================================================

def test_f1_score_empty_input():
    """1. Empty input raises ValueError."""
    with pytest.raises(ValueError):
        metrics.f1_score([], [])


def test_f1_score_mismatch_shapes():
    """2. Mismatching shapes raises ValueError."""
    with pytest.raises(ValueError):
        metrics.f1_score([1, 0], [1, 0, 1])


def test_f1_score_zero_division_default():
    """3. Default zero_division behavior returns 0.0 with warning when no true nor predicted samples."""
    with pytest.warns(UserWarning):
        score = metrics.f1_score([0, 0], [0, 0])
    assert score == 0.0


def test_f1_score_zero_division_custom():
    """4. zero_division parameter controls outputs for f1_score."""
    score_one = metrics.f1_score([0, 0], [0, 0], zero_division=1.0)
    assert score_one == 1.0
    
    score_nan = metrics.f1_score([0, 0], [0, 0], zero_division=np.nan)
    assert np.isnan(score_nan)


def test_f1_score_multiclass_missing_class():
    """5. Multi-class macro-average f1 when some classes have no instances."""
    y_true = [0, 1, 0]  # class 2 has no true instances
    y_pred = [0, 1, 2]  # class 2 has 1 predicted instance
    # F1 score macro average
    score = metrics.f1_score(y_true, y_pred, average="macro")
    # F1 for class 0: 2 * (1/2) * (1/1) / (1/2 + 1) = 2/3
    # F1 for class 1: 1.0
    # F1 for class 2: 0.0
    np.testing.assert_allclose(score, (2.0 / 3.0 + 1.0 + 0.0) / 3.0)


# =====================================================================
# roc_auc_score Boundary Cases
# =====================================================================

def test_roc_auc_score_empty_input():
    """1. Empty input raises ValueError."""
    with pytest.raises(ValueError):
        metrics.roc_auc_score([], [])


def test_roc_auc_score_mismatch_shapes():
    """2. Mismatching shapes raises ValueError."""
    with pytest.raises(ValueError):
        metrics.roc_auc_score([1, 0], [0.5, 0.5, 0.5])


def test_roc_auc_score_single_class_target():
    """3. Single class target returns NaN with warning."""
    with pytest.warns(UserWarning):
        score = metrics.roc_auc_score([1, 1, 1], [0.5, 0.6, 0.7])
    assert np.isnan(score)


def test_roc_auc_score_perfect_predictions():
    """4. Perfect predictions yield ROC AUC of 1.0."""
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.2, 0.8, 0.9]
    score = metrics.roc_auc_score(y_true, y_score)
    assert score == 1.0


def test_roc_auc_score_worse_than_random():
    """5. Inverted predictions yield ROC AUC of 0.0."""
    y_true = [0, 0, 1, 1]
    y_score = [0.9, 0.8, 0.2, 0.1]
    score = metrics.roc_auc_score(y_true, y_score)
    assert score == 0.0


# =====================================================================
# mean_squared_error Boundary Cases
# =====================================================================

def test_mean_squared_error_empty_input():
    """1. Empty input raises ValueError."""
    with pytest.raises(ValueError):
        metrics.mean_squared_error([], [])


def test_mean_squared_error_mismatch_shapes():
    """2. Mismatching shapes raises ValueError."""
    with pytest.raises(ValueError):
        metrics.mean_squared_error([1.0, 2.0], [1.0, 2.0, 3.0])


def test_mean_squared_error_single_sample():
    """3. Single sample input returns squared difference."""
    score = metrics.mean_squared_error([10.0], [7.0])
    np.testing.assert_allclose(score, 9.0)


def test_mean_squared_error_multioutput():
    """4. Multi-output regression raw vs average values."""
    y_true = [[1.0, 2.0], [3.0, 4.0]]
    y_pred = [[1.0, 4.0], [2.0, 4.0]]
    # squared diffs:
    # output 0: (0^2 + 1^2) / 2 = 0.5
    # output 1: (2^2 + 0^2) / 2 = 2.0
    raw = metrics.mean_squared_error(y_true, y_pred, multioutput="raw_values")
    np.testing.assert_array_almost_equal(raw, [0.5, 2.0])
    
    avg = metrics.mean_squared_error(y_true, y_pred, multioutput="uniform_average")
    assert avg == 1.25


def test_mean_squared_error_extreme_values():
    """5. Extremely large differences do not crash and return valid float."""
    y_true = [1e10, 0.0]
    y_pred = [0.0, 1e10]
    score = metrics.mean_squared_error(y_true, y_pred)
    assert np.isfinite(score)
    np.testing.assert_allclose(score, 1e20)


# =====================================================================
# r2_score Boundary Cases
# =====================================================================

def test_r2_score_empty_input():
    """1. Empty input raises ValueError."""
    with pytest.raises(ValueError):
        metrics.r2_score([], [])


def test_r2_score_mismatch_shapes():
    """2. Mismatching shapes raises ValueError."""
    with pytest.raises(ValueError):
        metrics.r2_score([1.0, 2.0], [1.0, 2.0, 3.0])


def test_r2_score_single_sample():
    """3. Single sample input returns nan with warning."""
    with pytest.warns(UserWarning):
        score = metrics.r2_score([5.0], [5.0])
    assert np.isnan(score)


def test_r2_score_zero_variance_target():
    """4. Zero variance target returns 1.0 if predictions match target, else 0.0."""
    score_perfect = metrics.r2_score([5.0, 5.0, 5.0], [5.0, 5.0, 5.0])
    assert score_perfect == 1.0
    
    score_imperfect = metrics.r2_score([5.0, 5.0, 5.0], [4.0, 5.0, 6.0])
    assert score_imperfect == 0.0


def test_r2_score_negative_r2():
    """5. Worse predictions than constant mean return negative R^2."""
    y_true = [1.0, 2.0, 3.0]
    y_pred = [3.0, 2.0, 1.0]
    score = metrics.r2_score(y_true, y_pred)
    assert score < 0.0
