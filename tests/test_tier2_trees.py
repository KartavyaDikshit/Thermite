import numpy as np
import pytest
from tests.conftest import get_module

# Dynamically import tree and ensemble modules
tree = get_module("tree")
ensemble = get_module("ensemble")


# =====================================================================
# DecisionTreeClassifier Boundary Cases
# =====================================================================

def test_decision_tree_classifier_empty_input():
    """1. Fitting on empty array raises ValueError."""
    dt = tree.DecisionTreeClassifier()
    with pytest.raises(ValueError):
        dt.fit(np.empty((0, 2)), np.empty(0))


def test_decision_tree_classifier_single_class():
    """2. Fitting with only 1 class in target should succeed, classes_ should have 1 element."""
    dt = tree.DecisionTreeClassifier()
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([5, 5, 5])
    
    dt.fit(X, y)
    assert list(dt.classes_) == [5]
    pred = dt.predict([[4.0]])
    np.testing.assert_array_equal(pred, [5])


def test_decision_tree_classifier_single_sample():
    """3. Fitting on a single sample should succeed and have depth 0."""
    dt = tree.DecisionTreeClassifier()
    X = np.array([[1.0, 2.0]])
    y = np.array([1])
    
    dt.fit(X, y)
    assert dt.get_depth() == 0
    pred = dt.predict([[2.0, 3.0]])
    np.testing.assert_array_equal(pred, [1])


def test_decision_tree_classifier_zero_variance_features():
    """4. Zero variance features should fit successfully."""
    dt = tree.DecisionTreeClassifier()
    X = np.array([[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]])
    y = np.array([0, 1, 0])
    
    dt.fit(X, y)
    assert dt.get_depth() == 0
    pred = dt.predict([[5.0, 5.0]])
    assert pred[0] in (0, 1)


def test_decision_tree_classifier_invalid_inputs():
    """5. Input containing inf raises ValueError."""
    dt = tree.DecisionTreeClassifier()
    with pytest.raises(ValueError):
        dt.fit([[1.0], [np.inf]], [0, 1])


# =====================================================================
# DecisionTreeRegressor Boundary Cases
# =====================================================================

def test_decision_tree_regressor_empty_input():
    """1. Fitting on empty array raises ValueError."""
    dt = tree.DecisionTreeRegressor()
    with pytest.raises(ValueError):
        dt.fit(np.empty((0, 2)), np.empty(0))


def test_decision_tree_regressor_single_sample():
    """2. Fitting on a single sample should succeed and predict that sample's target."""
    dt = tree.DecisionTreeRegressor()
    X = np.array([[1.0, 2.0]])
    y = np.array([10.5])
    
    dt.fit(X, y)
    pred = dt.predict([[3.0, 4.0]])
    np.testing.assert_allclose(pred, [10.5])


def test_decision_tree_regressor_zero_variance_features():
    """3. Zero variance features should fit successfully."""
    dt = tree.DecisionTreeRegressor()
    X = np.array([[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]])
    y = np.array([10.0, 20.0, 30.0])
    
    dt.fit(X, y)
    assert dt.get_depth() == 0
    pred = dt.predict([[5.0, 5.0]])
    np.testing.assert_allclose(pred, [20.0])  # mean of targets


def test_decision_tree_regressor_constant_target():
    """4. Constant target should result in depth 0."""
    dt = tree.DecisionTreeRegressor()
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([5.5, 5.5, 5.5])
    
    dt.fit(X, y)
    assert dt.get_depth() == 0
    pred = dt.predict([[4.0]])
    np.testing.assert_allclose(pred, [5.5])


def test_decision_tree_regressor_invalid_inputs():
    """5. Input containing inf raises ValueError."""
    dt = tree.DecisionTreeRegressor()
    with pytest.raises(ValueError):
        dt.fit([[1.0], [np.inf]], [1.0, 2.0])


# =====================================================================
# RandomForestClassifier Boundary Cases
# =====================================================================

def test_random_forest_classifier_empty_input():
    """1. Fitting on empty array raises ValueError."""
    rf = ensemble.RandomForestClassifier()
    with pytest.raises(ValueError):
        rf.fit(np.empty((0, 2)), np.empty(0))


def test_random_forest_classifier_single_class():
    """2. Fitting with only 1 class in target succeeds."""
    rf = ensemble.RandomForestClassifier()
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([1, 1, 1])
    
    rf.fit(X, y)
    assert list(rf.classes_) == [1]
    pred = rf.predict([[4.0]])
    np.testing.assert_array_equal(pred, [1])


def test_random_forest_classifier_single_sample():
    """3. Fitting on single sample should succeed."""
    rf = ensemble.RandomForestClassifier()
    X = np.array([[1.0, 2.0]])
    y = np.array([0])
    
    rf.fit(X, y)
    pred = rf.predict([[2.0, 3.0]])
    np.testing.assert_array_equal(pred, [0])


def test_random_forest_classifier_zero_variance_features():
    """4. Zero variance features should fit successfully."""
    rf = ensemble.RandomForestClassifier()
    X = np.array([[5.0, 5.0], [5.0, 5.0]])
    y = np.array([0, 1])
    
    rf.fit(X, y)
    pred = rf.predict([[5.0, 5.0]])
    assert pred[0] in (0, 1)


def test_random_forest_classifier_invalid_estimators():
    """5. Invalid parameter n_estimators=0 should raise ValueError."""
    with pytest.raises(ValueError):
        rf = ensemble.RandomForestClassifier(n_estimators=0)
        rf.fit([[1.0]], [1])


# =====================================================================
# RandomForestRegressor Boundary Cases
# =====================================================================

def test_random_forest_regressor_empty_input():
    """1. Fitting on empty array raises ValueError."""
    rf = ensemble.RandomForestRegressor()
    with pytest.raises(ValueError):
        rf.fit(np.empty((0, 2)), np.empty(0))


def test_random_forest_regressor_single_sample():
    """2. Fitting on single sample should succeed."""
    rf = ensemble.RandomForestRegressor()
    X = np.array([[1.0, 2.0]])
    y = np.array([1.5])
    
    rf.fit(X, y)
    pred = rf.predict([[2.0, 3.0]])
    np.testing.assert_allclose(pred, [1.5])


def test_random_forest_regressor_zero_variance_features():
    """3. Zero variance features should fit successfully."""
    rf = ensemble.RandomForestRegressor(bootstrap=False)
    X = np.array([[5.0, 5.0], [5.0, 5.0]])
    y = np.array([1.5, 2.5])
    
    rf.fit(X, y)
    pred = rf.predict([[5.0, 5.0]])
    np.testing.assert_allclose(pred, [2.0], atol=0.1)


def test_random_forest_regressor_constant_target():
    """4. Constant target should predict constant."""
    rf = ensemble.RandomForestRegressor()
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([4.2, 4.2, 4.2])
    
    rf.fit(X, y)
    pred = rf.predict([[5.0]])
    np.testing.assert_allclose(pred, [4.2], atol=0.1)


def test_random_forest_regressor_invalid_estimators():
    """5. Invalid parameter n_estimators=0 should raise ValueError."""
    with pytest.raises(ValueError):
        rf = ensemble.RandomForestRegressor(n_estimators=0)
        rf.fit([[1.0]], [1.0])


# =====================================================================
# GradientBoostingClassifier Boundary Cases
# =====================================================================

def test_gradient_boosting_classifier_empty_input():
    """1. Fitting on empty array raises ValueError."""
    gb = ensemble.GradientBoostingClassifier()
    with pytest.raises(ValueError):
        gb.fit(np.empty((0, 2)), np.empty(0))


def test_gradient_boosting_classifier_single_class():
    """2. Fitting with only 1 class in target raises ValueError."""
    gb = ensemble.GradientBoostingClassifier()
    X = np.array([[1.0], [2.0]])
    y = np.array([1, 1])
    
    with pytest.raises(ValueError):
        gb.fit(X, y)


def test_gradient_boosting_classifier_single_sample():
    """3. Fitting on single sample with 2 classes (impossible but checks shape error/class count error).
    We fit with two samples to ensure 2 classes, then check if we can predict single sample."""
    gb = ensemble.GradientBoostingClassifier()
    X = np.array([[1.0], [2.0]])
    y = np.array([0, 1])
    
    gb.fit(X, y)
    pred = gb.predict([[1.5]])
    assert pred.shape == (1,)


def test_gradient_boosting_classifier_zero_variance_features():
    """4. Zero variance features should fit successfully."""
    gb = ensemble.GradientBoostingClassifier()
    X = np.array([[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]])
    y = np.array([0, 1, 0])
    
    gb.fit(X, y)
    pred = gb.predict([[5.0, 5.0]])
    assert pred[0] in (0, 1)


def test_gradient_boosting_classifier_invalid_estimators():
    """5. Invalid parameter n_estimators=0 should raise ValueError."""
    with pytest.raises(ValueError):
        gb = ensemble.GradientBoostingClassifier(n_estimators=0)
        gb.fit([[1.0], [2.0]], [0, 1])


# =====================================================================
# GradientBoostingRegressor Boundary Cases
# =====================================================================

def test_gradient_boosting_regressor_empty_input():
    """1. Fitting on empty array raises ValueError."""
    gb = ensemble.GradientBoostingRegressor()
    with pytest.raises(ValueError):
        gb.fit(np.empty((0, 2)), np.empty(0))


def test_gradient_boosting_regressor_single_sample():
    """2. Fitting on single sample should succeed."""
    gb = ensemble.GradientBoostingRegressor()
    X = np.array([[1.0, 2.0]])
    y = np.array([10.5])
    
    gb.fit(X, y)
    pred = gb.predict([[2.0, 3.0]])
    np.testing.assert_allclose(pred, [10.5])


def test_gradient_boosting_regressor_zero_variance_features():
    """3. Zero variance features should fit successfully."""
    gb = ensemble.GradientBoostingRegressor()
    X = np.array([[5.0, 5.0], [5.0, 5.0]])
    y = np.array([1.5, 2.5])
    
    gb.fit(X, y)
    pred = gb.predict([[5.0, 5.0]])
    np.testing.assert_allclose(pred, [2.0])  # mean of targets


def test_gradient_boosting_regressor_constant_target():
    """4. Constant target should predict constant."""
    gb = ensemble.GradientBoostingRegressor()
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([4.2, 4.2, 4.2])
    
    gb.fit(X, y)
    pred = gb.predict([[5.0]])
    np.testing.assert_allclose(pred, [4.2])


def test_gradient_boosting_regressor_invalid_estimators():
    """5. Invalid parameter n_estimators=0 should raise ValueError."""
    with pytest.raises(ValueError):
        gb = ensemble.GradientBoostingRegressor(n_estimators=0)
        gb.fit([[1.0]], [1.0])
