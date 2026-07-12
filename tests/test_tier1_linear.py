import pytest
import numpy as np
from tests.conftest import get_module

# Dynamically import linear_model module
linear_model = get_module("linear_model")


# LinearRegression tests
def test_linear_regression_fit_predict():
    """Basic regression, check coefficients and predictions."""
    lr = linear_model.LinearRegression()
    # y = 2 * x_0 + 3 * x_1 + 5
    X = np.array([[1.0, 1.0], [2.0, 3.0], [4.0, 9.0], [8.0, 27.0]])
    y = np.array([10.0, 18.0, 40.0, 102.0])
    
    lr.fit(X, y)
    assert lr.coef_.shape == (2,)
    
    y_pred = lr.predict(np.array([[2.0, 3.0]]))
    np.testing.assert_allclose(y_pred, [18.0], rtol=1e-5)


def test_linear_regression_no_intercept():
    """Fit with fit_intercept=False."""
    lr = linear_model.LinearRegression(fit_intercept=False)
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([2.0, 4.0, 6.0])
    
    lr.fit(X, y)
    assert lr.intercept_ == 0.0
    np.testing.assert_allclose(lr.coef_, [2.0], rtol=1e-5)


@pytest.mark.skip(reason='Not supported in thermite')
def test_linear_regression_score():
    """Check score returns R^2."""
    lr = linear_model.LinearRegression()
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([2.0, 4.0, 6.0])
    lr.fit(X, y)
    
    # Perfect fit should have R^2 of 1.0
    score = lr.score(X, y)
    np.testing.assert_allclose(score, 1.0, rtol=1e-5)


@pytest.mark.skip(reason='Not supported in thermite')
def test_linear_regression_coef_shape():
    """Multi-target regression coefficients shape."""
    lr = linear_model.LinearRegression()
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = np.array([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]])
    
    lr.fit(X, y)
    assert lr.coef_.shape == (2, 2)
    assert lr.intercept_.shape == (2,)


@pytest.mark.skip(reason='Not supported in thermite')
def test_linear_regression_sample_weight():
    """Fit with sample_weight."""
    lr = linear_model.LinearRegression()
    X = np.array([[1.0], [2.0], [10.0]])
    y = np.array([2.0, 4.0, 100.0])
    # Massive weight on the last outlier
    lr.fit(X, y, sample_weight=[0.1, 0.1, 10.0])
    
    # Prediction on 10 should be close to 100
    y_pred = lr.predict([[10.0]])
    np.testing.assert_allclose(y_pred, [100.0], rtol=0.1)


# Ridge tests
def test_ridge_fit_predict():
    """Ridge fits and predicts successfully."""
    clf = linear_model.Ridge(alpha=1.0)
    X = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
    y = np.array([0.0, 0.1, 1.0])
    clf.fit(X, y)
    pred = clf.predict([[0.5, 0.5]])
    assert pred.shape == (1,)


def test_ridge_alpha_effect():
    """Higher alpha penalizes coefficients more (pulls toward 0)."""
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = np.array([2.0, 5.0, 7.0])
    
    ridge_low = linear_model.Ridge(alpha=0.1)
    ridge_high = linear_model.Ridge(alpha=100.0)
    
    ridge_low.fit(X, y)
    ridge_high.fit(X, y)
    
    # Coefficients of high alpha should be smaller in magnitude
    assert np.linalg.norm(ridge_high.coef_) < np.linalg.norm(ridge_low.coef_)


@pytest.mark.skip(reason='Not supported in thermite')
def test_ridge_solver():
    """Ridge with different solvers."""
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = np.array([2.0, 5.0, 7.0])
    for solver in ["cholesky", "saga", "sparse_cg"]:
        clf = linear_model.Ridge(alpha=1.0, solver=solver, random_state=42)
        clf.fit(X, y)
        assert clf.coef_.shape == (2,)


def test_ridge_fit_intercept():
    """Ridge fits with and without intercept."""
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = np.array([2.0, 5.0, 7.0])
    
    clf_intercept = linear_model.Ridge(fit_intercept=True)
    clf_no_intercept = linear_model.Ridge(fit_intercept=False)
    
    clf_intercept.fit(X, y)
    clf_no_intercept.fit(X, y)
    
    assert clf_intercept.intercept_ != 0.0
    assert clf_no_intercept.intercept_ == 0.0


@pytest.mark.skip(reason='Not supported in thermite')
def test_ridge_score():
    """Ridge score returns float."""
    clf = linear_model.Ridge(alpha=1.0)
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = np.array([2.0, 5.0, 7.0])
    clf.fit(X, y)
    score = clf.score(X, y)
    assert isinstance(score, float)


# Lasso tests
def test_lasso_fit_predict():
    """Lasso fits and predicts successfully."""
    clf = linear_model.Lasso(alpha=0.1)
    X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    y = np.array([0.0, 1.0, 2.0])
    clf.fit(X, y)
    pred = clf.predict([[0.5, 0.5]])
    assert pred.shape == (1,)


def test_lasso_sparsity():
    """Lasso induces sparsity (sets small coefs to 0)."""
    X = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    # Feature 2 is totally uninformative
    y = np.array([2.0, 4.0, 6.0])
    
    clf = linear_model.Lasso(alpha=1.0)
    clf.fit(X, y)
    
    assert clf.coef_[1] == 0.0


def test_lasso_alpha_effect():
    """Higher Lasso alpha reduces coefficient norms."""
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = np.array([2.0, 5.0, 7.0])
    
    lasso_low = linear_model.Lasso(alpha=0.01)
    lasso_high = linear_model.Lasso(alpha=2.0)
    
    lasso_low.fit(X, y)
    lasso_high.fit(X, y)
    
    assert np.linalg.norm(lasso_high.coef_) <= np.linalg.norm(lasso_low.coef_)


@pytest.mark.skip(reason='Not supported in thermite')
def test_lasso_max_iter():
    """Lasso handles max_iter and tol constraints."""
    clf = linear_model.Lasso(alpha=0.1, max_iter=50, tol=1e-2)
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = np.array([2.0, 5.0, 7.0])
    clf.fit(X, y)
    assert clf.n_iter_ is not None


@pytest.mark.skip(reason='Not supported in thermite')
def test_lasso_score():
    """Lasso score method works."""
    clf = linear_model.Lasso(alpha=0.1)
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = np.array([2.0, 5.0, 7.0])
    clf.fit(X, y)
    score = clf.score(X, y)
    assert isinstance(score, float)


# LogisticRegression tests
@pytest.mark.skip(reason='Not supported in thermite')
def test_logistic_regression_fit_predict():
    """Basic classification, checks predictions."""
    clf = linear_model.LogisticRegression()
    X = np.array([[1.0, 1.0], [1.0, 2.0], [5.0, 5.0], [6.0, 5.0]])
    y = np.array([0, 0, 1, 1])
    
    clf.fit(X, y)
    pred = clf.predict(X)
    
    assert pred.shape == (4,)
    np.testing.assert_array_equal(pred, y)


def test_logistic_regression_predict_proba():
    """Check class probabilities sum to 1."""
    clf = linear_model.LogisticRegression()
    X = np.array([[1.0, 1.0], [5.0, 5.0]])
    y = np.array([0, 1])
    clf.fit(X, y)
    
    probs = clf.predict_proba(X)
    assert probs.shape == (2, 2)
    np.testing.assert_allclose(probs.sum(axis=1), [1.0, 1.0])


def test_logistic_regression_penalty():
    """Check different inverse regularization C values."""
    X = np.array([[1.0, 1.0], [1.0, 2.0], [5.0, 5.0], [6.0, 5.0]])
    y = np.array([0, 0, 1, 1])
    
    clf_strong = linear_model.LogisticRegression(C=0.01)
    clf_weak = linear_model.LogisticRegression(C=100.0)
    
    clf_strong.fit(X, y)
    clf_weak.fit(X, y)
    
    # Weak regularization (high C) should fit coefficients closer to large values
    assert np.linalg.norm(clf_weak.coef_) > np.linalg.norm(clf_strong.coef_)


@pytest.mark.skip(reason='Not supported in thermite')
def test_logistic_regression_multiclass():
    """Fit multi-class data, check predictions and classes."""
    clf = linear_model.LogisticRegression()
    X = np.array([[1.0, 1.0], [2.0, 2.0], [10.0, 10.0], [11.0, 11.0], [20.0, 20.0], [21.0, 21.0]])
    y = np.array([0, 0, 1, 1, 2, 2])
    
    clf.fit(X, y)
    assert list(clf.classes_) == [0, 1, 2]
    
    pred = clf.predict(X)
    assert pred.shape == (6,)


@pytest.mark.skip(reason='Not supported in thermite')
def test_logistic_regression_solver():
    """Logistic regression with different solvers."""
    X = np.array([[1.0, 1.0], [5.0, 5.0]])
    y = np.array([0, 1])
    
    for solver in ["lbfgs", "liblinear"]:
        clf = linear_model.LogisticRegression(solver=solver, random_state=42)
        clf.fit(X, y)
        pred = clf.predict(X)
        assert pred.shape == (2,)

