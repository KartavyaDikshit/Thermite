"""
tests/test_phase4_polars.py
Phase 4: Polars integration and device API tests.
"""

import numpy as np
import pytest

# ------------------------------------------------------------------ #
# Polars optional
# ------------------------------------------------------------------ #
try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

polars_only = pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")


# ------------------------------------------------------------------ #
# device.py tests (always available)
# ------------------------------------------------------------------ #
from thermite.device import validate_device, is_gpu, DEVICE_CPU, DEVICE_GPU, DEVICE_CUDA

def test_device_validate_cpu():
    assert validate_device("cpu") == "cpu"

def test_device_validate_gpu():
    assert validate_device("gpu") == "gpu"

def test_device_validate_cuda_alias():
    assert validate_device("cuda") == "cuda"

def test_device_is_gpu_true():
    assert is_gpu("gpu") is True
    assert is_gpu("cuda") is True

def test_device_is_gpu_false():
    assert is_gpu("cpu") is False

def test_device_invalid():
    with pytest.raises(ValueError):
        validate_device("tpu")


# ------------------------------------------------------------------ #
# Polars compat tests
# ------------------------------------------------------------------ #
@polars_only
def test_from_polars_X_basic():
    from thermite.polars_compat import from_polars_X
    df = pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    X = from_polars_X(df)
    assert isinstance(X, np.ndarray)
    assert X.shape == (2, 2)
    assert X.dtype == np.float64

@polars_only
def test_from_polars_y_basic():
    from thermite.polars_compat import from_polars_y
    s = pl.Series("label", [0, 1, 0, 1])
    y = from_polars_y(s)
    assert isinstance(y, np.ndarray)
    assert y.shape == (4,)
    assert y.dtype == np.float64

@polars_only
def test_from_polars_split():
    from thermite.polars_compat import from_polars
    df = pl.DataFrame({
        "feat1": [1.0, 2.0, 3.0],
        "feat2": [4.0, 5.0, 6.0],
        "target": [0.0, 1.0, 0.0],
    })
    X, y = from_polars(df, target_col="target")
    assert X.shape == (3, 2)
    assert y.shape == (3,)
    np.testing.assert_array_equal(y, [0.0, 1.0, 0.0])

@polars_only
def test_make_polars_pipeline_fit_predict():
    from thermite.polars_compat import make_polars_pipeline
    from thermite.linear_model import LogisticRegression

    df_train = pl.DataFrame({
        "x1": [0.1, 0.2, 0.9, 0.8, 0.1, 0.9],
        "x2": [0.1, 0.2, 0.8, 0.9, 0.2, 0.8],
        "y":  [0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
    })
    X_train = df_train.select(["x1", "x2"])
    y_train = df_train["y"]

    model = make_polars_pipeline(LogisticRegression(max_iter=500))
    model.fit(X_train, y_train)

    df_test = pl.DataFrame({"x1": [0.1, 0.9], "x2": [0.2, 0.8]})
    preds = model.predict(df_test)
    assert preds.shape == (2,)

@polars_only
def test_from_polars_X_wrong_type():
    from thermite.polars_compat import from_polars_X
    with pytest.raises(TypeError):
        from_polars_X([[1, 2], [3, 4]])

@polars_only
def test_from_polars_y_wrong_type():
    from thermite.polars_compat import from_polars_y
    with pytest.raises(TypeError):
        from_polars_y([1, 2, 3])


# ------------------------------------------------------------------ #
# partial_fit for LogisticRegression
# ------------------------------------------------------------------ #
from thermite.linear_model import LogisticRegression

def test_partial_fit_lr_convergence():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(1000, 10))
    y = (X[:, 0] + X[:, 1] > 0).astype(float)

    clf = LogisticRegression(C=1.0)
    for i in range(0, 1000, 100):
        clf.partial_fit(X[i:i+100], y[i:i+100], classes=[0.0, 1.0])

    acc = (clf.predict(X) == y).mean()
    assert acc > 0.8, f"Expected >80% acc, got {acc:.3f}"

def test_partial_fit_lr_incremental_classes():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))
    y = np.where(X[:, 0] > 0, 1.0, 0.0)

    clf = LogisticRegression()
    clf.partial_fit(X, y, classes=[0.0, 1.0])
    assert clf.coef_ is not None
    assert clf.predict(X).shape == (200,)


# ------------------------------------------------------------------ #
# GPU device on RandomForest
# ------------------------------------------------------------------ #
from thermite.ensemble import RandomForestClassifier, RandomForestRegressor

def test_rf_cpu_device():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(200, 10))
    y = (X[:, 0] > 0).astype(float)
    clf = RandomForestClassifier(n_estimators=20, random_state=42, device='cpu')
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (200,)

def test_rf_gpu_device_identical_output():
    """GPU device path (CPU fallback) must produce identical results."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(200, 10))
    y = (X[:, 0] > 0).astype(float)

    clf_cpu = RandomForestClassifier(n_estimators=20, random_state=42, device='cpu')
    clf_gpu = RandomForestClassifier(n_estimators=20, random_state=42, device='gpu')

    clf_cpu.fit(X, y)
    clf_gpu.fit(X, y)

    np.testing.assert_array_equal(clf_cpu.predict(X), clf_gpu.predict(X))

def test_rfr_gpu_device():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(200, 10))
    y = X[:, 0] * 2.0 + rng.normal(scale=0.1, size=200)
    rfr = RandomForestRegressor(n_estimators=20, random_state=7, device='gpu')
    rfr.fit(X, y)
    preds = rfr.predict(X)
    ss_res = np.sum((y - preds)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - ss_res / ss_tot
    assert r2 > 0.8, f"Expected R2>0.8, got {r2:.3f}"
