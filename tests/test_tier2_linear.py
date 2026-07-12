import numpy as np
import pytest
from tests.conftest import get_module

# Dynamically import linear_model module
linear_model = get_module("linear_model")


# =====================================================================
# LinearRegression Boundary Cases
# =====================================================================

def test_linear_regression_empty_input():
    """1. Fitting on empty array raises ValueError."""
    lr = linear_model.LinearRegression()
    with pytest.raises(ValueError):
        lr.fit(np.empty((0, 2)), np.empty(0))


def test_linear_regression_underdetermined():
    """2. Fitting underdetermined system (fewer samples than features) should fit perfectly."""
    lr = linear_model.LinearRegression()
    X = np.array([[1.0, 2.0, 3.0]])
    y = np.array([5.0])
    lr.fit(X, y)
    
    # Prediction on training sample should be exact (within precision)
    pred = lr.predict(X)
    np.testing.assert_allclose(pred, y, rtol=1e-5)


def test_linear_regression_perfect_collinearity():
    """3. Fitting on perfectly collinear features should not crash and should predict correctly."""
    lr = linear_model.LinearRegression()
    X = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    y = np.array([2.0, 4.0, 6.0])
    
    lr.fit(X, y)
    pred = lr.predict(X)
    np.testing.assert_allclose(pred, y, rtol=1e-5)


def test_linear_regression_zero_variance_target():
    """4. Fitting with zero variance target (all y values identical) should result in 0 coefficients."""
    lr = linear_model.LinearRegression()
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([7.0, 7.0, 7.0])
    
    lr.fit(X, y)
    np.testing.assert_allclose(lr.coef_, [0.0], atol=1e-7)
    np.testing.assert_allclose(lr.intercept_, 7.0, rtol=1e-5)


def test_linear_regression_extreme_inputs():
    """5. Extremely large/small values should not cause numerical overflow/underflow errors."""
    lr = linear_model.LinearRegression()
    X = np.array([[1e10], [-1e10]])
    y = np.array([1e10, -1e10])
    
    lr.fit(X, y)
    pred = lr.predict(X)
    np.testing.assert_allclose(pred, y, rtol=1e-5)
    np.testing.assert_allclose(lr.coef_, [1.0], rtol=1e-5)


# =====================================================================
# Ridge Boundary Cases
# =====================================================================

def test_ridge_empty_input():
    """1. Fitting on empty array raises ValueError."""
    clf = linear_model.Ridge()
    with pytest.raises(ValueError):
        clf.fit(np.empty((0, 2)), np.empty(0))


def test_ridge_underdetermined():
    """2. Fitting underdetermined system (fewer samples than features) is stabilized by L2 penalty."""
    clf = linear_model.Ridge(alpha=1.0)
    X = np.array([[1.0, 2.0, 3.0]])
    y = np.array([5.0])
    clf.fit(X, y)
    
    assert clf.coef_.shape == (3,)
    assert np.all(np.isfinite(clf.coef_))
    # Should not crash and make a prediction
    pred = clf.predict([[2.0, 3.0, 4.0]])
    assert np.isfinite(pred[0])


def test_ridge_perfect_collinearity():
    """3. Ridge handles perfect collinearity by distributing weights/regularizing."""
    clf = linear_model.Ridge(alpha=1.0)
    X = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    y = np.array([2.0, 4.0, 6.0])
    
    clf.fit(X, y)
    pred = clf.predict(X)
    assert np.all(np.isfinite(pred))


def test_ridge_zero_variance_target():
    """4. Ridge with zero variance target."""
    clf = linear_model.Ridge(alpha=1.0)
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([10.0, 10.0, 10.0])
    
    clf.fit(X, y)
    np.testing.assert_allclose(clf.coef_, [0.0], atol=1e-7)
    np.testing.assert_allclose(clf.intercept_, 10.0, rtol=1e-5)


def test_ridge_extremely_large_alpha():
    """5. Extremely large alpha should drive coefficients very close to 0."""
    clf = linear_model.Ridge(alpha=1e10)
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = np.array([2.0, 5.0, 7.0])
    
    clf.fit(X, y)
    np.testing.assert_allclose(clf.coef_, [0.0, 0.0], atol=1e-5)


# =====================================================================
# Lasso Boundary Cases
# =====================================================================

def test_lasso_empty_input():
    """1. Fitting on empty array raises ValueError."""
    clf = linear_model.Lasso()
    with pytest.raises(ValueError):
        clf.fit(np.empty((0, 2)), np.empty(0))


def test_lasso_zero_variance_target():
    """2. Lasso with zero variance target should set coefs to 0."""
    clf = linear_model.Lasso(alpha=0.1)
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([5.0, 5.0, 5.0])
    
    clf.fit(X, y)
    np.testing.assert_allclose(clf.coef_, [0.0], atol=1e-7)
    np.testing.assert_allclose(clf.intercept_, 5.0, rtol=1e-5)


def test_lasso_extremely_large_alpha():
    """3. Extremely large alpha should set all coefficients to exactly 0."""
    clf = linear_model.Lasso(alpha=1e5)
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = np.array([2.0, 5.0, 7.0])
    
    clf.fit(X, y)
    np.testing.assert_allclose(clf.coef_, [0.0, 0.0], atol=1e-7)


def test_lasso_perfect_collinearity():
    """4. Lasso selects one of the collinear features and drops the other."""
    clf = linear_model.Lasso(alpha=0.1)
    X = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    y = np.array([2.0, 4.0, 6.0])
    
    clf.fit(X, y)
    # One coefficient should be non-zero (or both regularized but sum up, sklearn Lasso typically yields one zeroed out)
    assert clf.coef_[0] == 0.0 or clf.coef_[1] == 0.0


def test_lasso_high_dimension():
    """5. Lasso with fewer samples than features runs and selects at most n_samples features."""
    clf = linear_model.Lasso(alpha=0.1)
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y = np.array([10.0, 20.0])
    
    clf.fit(X, y)
    # Fit should succeed and predict should work
    pred = clf.predict(X)
    assert pred.shape == (2,)


# =====================================================================
# LogisticRegression Boundary Cases
# =====================================================================

def test_logistic_regression_empty_input():
    """1. Fitting on empty array raises ValueError."""
    clf = linear_model.LogisticRegression()
    with pytest.raises(ValueError):
        clf.fit(np.empty((0, 2)), np.empty(0))


def test_logistic_regression_single_class_target():
    """2. Fitting with only 1 class in target raises ValueError."""
    clf = linear_model.LogisticRegression()
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([1, 1])
    
    with pytest.raises(ValueError):
        clf.fit(X, y)


def test_logistic_regression_perfectly_separable():
    """3. Logistic Regression on perfectly separable classes should fit and predict correctly."""
    clf = linear_model.LogisticRegression()
    X = np.array([[-10.0], [10.0]])
    y = np.array([0, 1])
    
    clf.fit(X, y)
    pred = clf.predict(X)
    np.testing.assert_array_equal(pred, y)


def test_logistic_regression_collinear_features():
    """4. Logistic Regression on perfectly collinear features should not crash."""
    clf = linear_model.LogisticRegression()
    X = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]])
    y = np.array([0, 0, 1, 1])
    
    clf.fit(X, y)
    assert clf.coef_.shape == (1, 2)


def test_logistic_regression_regularization_extremes():
    """5. Testing C parameter extremes (very small C driving coefs to 0, large C allowing them to grow)."""
    X = np.array([[-1.0, -1.0], [1.0, 1.0], [-2.0, -1.0], [2.0, 2.0]])
    y = np.array([0, 1, 0, 1])
    
    clf_strong = linear_model.LogisticRegression(C=1e-5)
    clf_weak = linear_model.LogisticRegression(C=1e5)
    
    clf_strong.fit(X, y)
    clf_weak.fit(X, y)
    
    assert np.linalg.norm(clf_strong.coef_) < np.linalg.norm(clf_weak.coef_)
