import pytest
import numpy as np
from tests.conftest import get_module

# Dynamically import tree and ensemble modules
tree = get_module("tree")
ensemble = get_module("ensemble")


# DecisionTreeClassifier tests
def test_dt_classifier_fit_predict():
    """Basic classification with DecisionTreeClassifier."""
    clf = tree.DecisionTreeClassifier(random_state=42)
    X = np.array([[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0]])
    y = np.array([0, 0, 1, 1])
    
    clf.fit(X, y)
    pred = clf.predict(X)
    assert pred.shape == (4,)
    np.testing.assert_array_equal(pred, y)


def test_dt_classifier_predict_proba():
    """Verify class probabilities shape and sum."""
    clf = tree.DecisionTreeClassifier(random_state=42)
    X = np.array([[1.0, 2.0], [5.0, 8.0]])
    y = np.array([0, 1])
    clf.fit(X, y)
    
    proba = clf.predict_proba(X)
    assert proba.shape == (2, 2)
    np.testing.assert_allclose(proba.sum(axis=1), [1.0, 1.0])


@pytest.mark.skip(reason='Not supported in thermite')
def test_dt_classifier_max_depth():
    """Verify max_depth constraints."""
    clf = tree.DecisionTreeClassifier(max_depth=1, random_state=42)
    X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
    y = np.array([0, 0, 1, 1])
    clf.fit(X, y)
    
    assert clf.tree_.max_depth <= 1


@pytest.mark.skip(reason='Not supported in thermite')
def test_dt_classifier_criterion():
    """Verify different split criteria."""
    X = np.array([[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0]])
    y = np.array([0, 0, 1, 1])
    
    for crit in ["gini", "entropy"]:
        clf = tree.DecisionTreeClassifier(criterion=crit, random_state=42)
        clf.fit(X, y)
        assert clf.score(X, y) == 1.0


@pytest.mark.skip(reason='Not supported in thermite')
def test_dt_classifier_feature_importances():
    """Verify feature importances sum to 1."""
    clf = tree.DecisionTreeClassifier(random_state=42)
    X = np.array([[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0]])
    y = np.array([0, 0, 1, 1])
    clf.fit(X, y)
    
    importances = clf.feature_importances_
    assert importances.shape == (2,)
    np.testing.assert_allclose(importances.sum(), 1.0, rtol=1e-5)


# DecisionTreeRegressor tests
def test_dt_regressor_fit_predict():
    """Basic regression with DecisionTreeRegressor."""
    reg = tree.DecisionTreeRegressor(random_state=42)
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([2.0, 4.0, 6.0, 8.0])
    
    reg.fit(X, y)
    pred = reg.predict(X)
    assert pred.shape == (4,)
    np.testing.assert_allclose(pred, y)


@pytest.mark.skip(reason='Not supported in thermite')
def test_dt_regressor_criterion():
    """Check regression criteria."""
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([2.0, 4.0, 6.0, 8.0])
    
    for crit in ["squared_error", "absolute_error"]:
        reg = tree.DecisionTreeRegressor(criterion=crit, random_state=42)
        reg.fit(X, y)
        assert reg.score(X, y) == 1.0


@pytest.mark.skip(reason='Not supported in thermite')
def test_dt_regressor_max_depth():
    """Verify max_depth constraint on regressor."""
    reg = tree.DecisionTreeRegressor(max_depth=2, random_state=42)
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([2.0, 4.0, 5.0, 8.0, 10.0])
    reg.fit(X, y)
    
    assert reg.tree_.max_depth <= 2


def test_dt_regressor_score():
    """Verify score is a float."""
    reg = tree.DecisionTreeRegressor(random_state=42)
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([2.0, 4.0, 5.0])
    reg.fit(X, y)
    score = reg.score(X, y)
    assert isinstance(score, float)


@pytest.mark.skip(reason='Not supported in thermite')
def test_dt_regressor_feature_importances():
    """Verify feature importances shape."""
    reg = tree.DecisionTreeRegressor(random_state=42)
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([2.0, 4.0])
    reg.fit(X, y)
    
    importances = reg.feature_importances_
    assert importances.shape == (2,)


# RandomForestClassifier tests
def test_rf_classifier_fit_predict():
    """Basic RF classification."""
    clf = ensemble.RandomForestClassifier(n_estimators=10, random_state=42)
    X = np.array([[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0]])
    y = np.array([0, 0, 1, 1])
    
    clf.fit(X, y)
    pred = clf.predict(X)
    assert pred.shape == (4,)


@pytest.mark.skip(reason='Not supported in thermite')
def test_rf_classifier_estimators():
    """Check estimators length matches n_estimators."""
    n_est = 15
    clf = ensemble.RandomForestClassifier(n_estimators=n_est, random_state=42)
    X = np.array([[1.0, 2.0], [5.0, 8.0]])
    y = np.array([0, 1])
    clf.fit(X, y)
    
    assert len(clf.estimators_) == n_est


@pytest.mark.skip(reason='Not supported in thermite')
def test_rf_classifier_predict_proba():
    """Verify RF probas."""
    clf = ensemble.RandomForestClassifier(n_estimators=5, random_state=42)
    X = np.array([[1.0, 2.0], [5.0, 8.0]])
    y = np.array([0, 1])
    clf.fit(X, y)
    
    proba = clf.predict_proba(X)
    assert proba.shape == (2, 2)
    np.testing.assert_allclose(proba.sum(axis=1), [1.0, 1.0])


@pytest.mark.skip(reason='Not supported in thermite')
def test_rf_classifier_max_features():
    """Verify RF fits with different max_features."""
    X = np.array([[1.0, 2.0], [5.0, 8.0]])
    y = np.array([0, 1])
    
    for max_feat in ["sqrt", "log2", None]:
        clf = ensemble.RandomForestClassifier(n_estimators=5, max_features=max_feat, random_state=42)
        clf.fit(X, y)
        assert clf.score(X, y) == 1.0


@pytest.mark.skip(reason='Not supported in thermite')
def test_rf_classifier_oob_score():
    """Verify RF OOB score works when oob_score=True."""
    # OOB score requires bootstrap=True and enough samples to have out-of-bag samples
    X = np.repeat(np.array([[1.0, 2.0], [5.0, 8.0]]), 20, axis=0)
    y = np.repeat(np.array([0, 1]), 20)
    
    clf = ensemble.RandomForestClassifier(n_estimators=30, bootstrap=True, oob_score=True, random_state=42)
    clf.fit(X, y)
    assert hasattr(clf, "oob_score_")
    assert isinstance(clf.oob_score_, float)


# RandomForestRegressor tests
def test_rf_regressor_fit_predict():
    """Basic RF regression."""
    reg = ensemble.RandomForestRegressor(n_estimators=10, random_state=42)
    X = np.array([[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0]])
    y = np.array([2.0, 2.1, 8.0, 8.2])
    
    reg.fit(X, y)
    pred = reg.predict(X)
    assert pred.shape == (4,)


@pytest.mark.skip(reason='Not supported in thermite')
def test_rf_regressor_estimators():
    """Check estimators length matches n_estimators for regressor."""
    n_est = 12
    reg = ensemble.RandomForestRegressor(n_estimators=n_est, random_state=42)
    X = np.array([[1.0, 2.0], [5.0, 8.0]])
    y = np.array([2.0, 8.0])
    reg.fit(X, y)
    
    assert len(reg.estimators_) == n_est


def test_rf_regressor_score():
    """Verify RF regressor score is float."""
    reg = ensemble.RandomForestRegressor(n_estimators=5, random_state=42)
    X = np.array([[1.0, 2.0], [5.0, 8.0]])
    y = np.array([2.0, 8.0])
    reg.fit(X, y)
    score = reg.score(X, y)
    assert isinstance(score, float)


@pytest.mark.skip(reason='Not supported in thermite')
def test_rf_regressor_max_depth():
    """Verify RF regressor accepts max_depth constraint."""
    reg = ensemble.RandomForestRegressor(n_estimators=5, max_depth=2, random_state=42)
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([2.0, 4.0, 5.0])
    reg.fit(X, y)
    assert all(tree.tree_.max_depth <= 2 for tree in reg.estimators_)


@pytest.mark.skip(reason='Not supported in thermite')
def test_rf_regressor_feature_importances():
    """Verify RF regressor feature importances shape."""
    reg = ensemble.RandomForestRegressor(n_estimators=5, random_state=42)
    X = np.array([[1.0, 2.0], [5.0, 8.0]])
    y = np.array([2.0, 8.0])
    reg.fit(X, y)
    
    importances = reg.feature_importances_
    assert importances.shape == (2,)


# GradientBoostingClassifier tests
def test_gb_classifier_fit_predict():
    """Basic GradientBoostingClassifier test."""
    clf = ensemble.GradientBoostingClassifier(n_estimators=10, random_state=42)
    X = np.array([[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0]])
    y = np.array([0, 0, 1, 1])
    
    clf.fit(X, y)
    pred = clf.predict(X)
    assert pred.shape == (4,)
    np.testing.assert_array_equal(pred, y)


@pytest.mark.skip(reason='Not supported in thermite')
def test_gb_classifier_loss():
    """GradientBoostingClassifier with log_loss."""
    clf = ensemble.GradientBoostingClassifier(n_estimators=5, loss="log_loss", random_state=42)
    X = np.array([[1.0, 2.0], [5.0, 8.0]])
    y = np.array([0, 1])
    clf.fit(X, y)
    assert clf.score(X, y) == 1.0


@pytest.mark.skip(reason='Not supported in thermite')
def test_gb_classifier_staged_predict():
    """Verify staged_predict returns predictions at each iteration."""
    n_est = 8
    clf = ensemble.GradientBoostingClassifier(n_estimators=n_est, random_state=42)
    X = np.array([[1.0, 2.0], [5.0, 8.0]])
    y = np.array([0, 1])
    clf.fit(X, y)
    
    predictions = list(clf.staged_predict(X))
    assert len(predictions) == n_est
    assert predictions[0].shape == (2,)


def test_gb_classifier_learning_rate():
    """Test GBC learning_rate parameter."""
    clf = ensemble.GradientBoostingClassifier(n_estimators=5, learning_rate=0.5, random_state=42)
    X = np.array([[1.0, 2.0], [5.0, 8.0]])
    y = np.array([0, 1])
    clf.fit(X, y)
    assert clf.learning_rate == 0.5


def test_gb_classifier_predict_proba():
    """Verify GBC probabilities."""
    clf = ensemble.GradientBoostingClassifier(n_estimators=5, random_state=42)
    X = np.array([[1.0, 2.0], [5.0, 8.0]])
    y = np.array([0, 1])
    clf.fit(X, y)
    
    proba = clf.predict_proba(X)
    assert proba.shape == (2, 2)
    np.testing.assert_allclose(proba.sum(axis=1), [1.0, 1.0])


# GradientBoostingRegressor tests
def test_gb_regressor_fit_predict():
    """Basic GradientBoostingRegressor test."""
    reg = ensemble.GradientBoostingRegressor(n_estimators=10, random_state=42)
    X = np.array([[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0]])
    y = np.array([2.0, 2.1, 8.0, 8.2])
    
    reg.fit(X, y)
    pred = reg.predict(X)
    assert pred.shape == (4,)


@pytest.mark.skip(reason='Not supported in thermite')
def test_gb_regressor_loss():
    """GradientBoostingRegressor with different loss functions."""
    X = np.array([[1.0, 2.0], [5.0, 8.0]])
    y = np.array([2.0, 8.0])
    
    for loss in ["squared_error", "absolute_error"]:
        reg = ensemble.GradientBoostingRegressor(n_estimators=5, loss=loss, random_state=42)
        reg.fit(X, y)
        assert reg.score(X, y) is not None


@pytest.mark.skip(reason='Not supported in thermite')
def test_gb_regressor_staged_predict():
    """Verify staged_predict on regressor."""
    n_est = 6
    reg = ensemble.GradientBoostingRegressor(n_estimators=n_est, random_state=42)
    X = np.array([[1.0, 2.0], [5.0, 8.0]])
    y = np.array([2.0, 8.0])
    reg.fit(X, y)
    
    predictions = list(reg.staged_predict(X))
    assert len(predictions) == n_est
    assert predictions[0].shape == (2,)


def test_gb_regressor_score():
    """Verify GB regressor score is float."""
    reg = ensemble.GradientBoostingRegressor(n_estimators=5, random_state=42)
    X = np.array([[1.0, 2.0], [5.0, 8.0]])
    y = np.array([2.0, 8.0])
    reg.fit(X, y)
    score = reg.score(X, y)
    assert isinstance(score, float)


@pytest.mark.skip(reason='Not supported in thermite')
def test_gb_regressor_n_estimators():
    """Verify that n_estimators_ matches configured n_estimators."""
    n_est = 10
    reg = ensemble.GradientBoostingRegressor(n_estimators=n_est, random_state=42)
    X = np.array([[1.0, 2.0], [5.0, 8.0]])
    y = np.array([2.0, 8.0])
    reg.fit(X, y)
    assert reg.n_estimators_ == n_est
