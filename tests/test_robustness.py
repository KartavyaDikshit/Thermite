import numpy as np
import pytest
from thermite.linear_model import LogisticRegression, LinearRegression
from thermite.tree import DecisionTreeClassifier, DecisionTreeRegressor
from thermite.cluster import KMeans
from thermite.svm import SVC
from thermite import _core

def test_sliced_arrays_robustness():
    # Generate simple dataset
    X = np.random.randn(100, 5)
    y = np.random.choice([0, 1], size=100).astype(np.float64)
    
    # Sliced (non-contiguous) arrays
    X_sliced = X[::2]
    y_sliced = y[::2]
    
    # Ensure they are indeed non-contiguous
    assert not X_sliced.flags['C_CONTIGUOUS']
    assert not y_sliced.flags['C_CONTIGUOUS']
    
    # Test LogisticRegression
    clf_lr = LogisticRegression()
    clf_lr.fit(X_sliced, y_sliced)
    preds_lr = clf_lr.predict(X_sliced)
    assert preds_lr.shape == (50,)
    
    # Test DecisionTreeClassifier
    clf_dt = DecisionTreeClassifier()
    clf_dt.fit(X_sliced, y_sliced)
    preds_dt = clf_dt.predict(X_sliced)
    assert preds_dt.shape == (50,)
    
    # Test KMeans
    clf_km = KMeans(n_clusters=2)
    clf_km.fit(X_sliced)
    preds_km = clf_km.predict(X_sliced)
    assert preds_km.shape == (50,)

    # Test SVC
    clf_svc = SVC(probability=True)
    clf_svc.fit(X_sliced, y_sliced)
    preds_svc = clf_svc.predict(X_sliced)
    assert preds_svc.shape == (50,)
    probas_svc = clf_svc.predict_proba(X_sliced)
    assert probas_svc.shape == (50, 2)


def test_svc_feature_mismatch_validation():
    X = np.random.randn(10, 5)
    y = np.random.choice([0, 1], size=10).astype(np.float64)
    
    clf = SVC(probability=True)
    clf.fit(X, y)
    
    # Predict with mismatched features (e.g., 4 instead of 5)
    X_mismatched = np.random.randn(5, 4)
    
    with pytest.raises(ValueError) as excinfo:
        clf.predict(X_mismatched)
    assert "features" in str(excinfo.value)
    
    with pytest.raises(ValueError) as excinfo:
        clf.predict_proba(X_mismatched)
    assert "features" in str(excinfo.value)


def test_pyo3_backend_raises_error_on_non_contiguous_array():
    # DecisionTreeClassifier fit takes X: PyReadonlyArray2, y: PyReadonlyArray1
    # If we pass y as a sliced array directly, it should fail.
    model = _core.DecisionTreeClassifier()
    X = np.random.randn(10, 5)
    y = np.random.choice([0, 1], size=10).astype(np.float64)
    
    # Slice y to make it non-contiguous
    y_non_contiguous = y[::2]
    # We also need a matching X size, which we can make contiguous
    X_sliced = np.ascontiguousarray(X[::2])
    
    with pytest.raises(ValueError) as excinfo:
        model.fit(X_sliced, y_non_contiguous, None)
    assert "Array must be contiguous" in str(excinfo.value)
