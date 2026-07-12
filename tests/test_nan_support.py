import pytest
import numpy as np
from tests.conftest import get_module

tree = get_module("tree")
linear_model = get_module("linear_model")

def test_nan_support_dt_classifier():
    rng = np.random.RandomState(42)
    X = rng.randn(200, 2)
    y = (X[:, 0] > 0.0).astype(float)
    
    # Introduce NaNs in 5% of the data
    mask = rng.rand(*X.shape) < 0.05
    X[mask] = np.nan

    clf = tree.DecisionTreeClassifier(random_state=42)
    clf.fit(X, y)
    preds = clf.predict(X)
    
    assert preds.shape == (200,)
    accuracy = np.mean(preds == y)
    print(f"DecisionTreeClassifier accuracy: {accuracy:.4f}")
    assert accuracy > 0.90

def test_nan_support_logistic_regression():
    rng = np.random.RandomState(42)
    X = rng.randn(200, 2)
    y = (X[:, 0] > 0.0).astype(float)
    
    # Introduce NaNs in 5% of the data
    mask = rng.rand(*X.shape) < 0.05
    X[mask] = np.nan

    clf = linear_model.LogisticRegression(C=1.0, penalty="l2")
    clf.fit(X, y)
    preds = clf.predict(X)
    
    assert preds.shape == (200,)
    accuracy = np.mean(preds == y)
    print(f"LogisticRegression accuracy: {accuracy:.4f}")
    assert accuracy > 0.90
