import numpy as np
import pytest
from tests.conftest import get_module

# Dynamically import preprocessing module
preprocessing = get_module("preprocessing")


# =====================================================================
# StandardScaler Boundary Cases
# =====================================================================

def test_standard_scaler_empty_input():
    """1. Fitting on an empty array should raise ValueError."""
    scaler = preprocessing.StandardScaler()
    with pytest.raises(ValueError):
        scaler.fit(np.empty((0, 3)))


def test_standard_scaler_single_row():
    """2. Fitting on a single-row input should set variance/std to 0 and scale to 1.0."""
    scaler = preprocessing.StandardScaler()
    X = np.array([[10.0, -5.0, 0.0]])
    scaler.fit(X)
    
    np.testing.assert_array_almost_equal(scaler.mean_, [10.0, -5.0, 0.0])
    np.testing.assert_array_almost_equal(scaler.var_, [0.0, 0.0, 0.0])
    np.testing.assert_array_almost_equal(scaler.scale_, [1.0, 1.0, 1.0])
    
    # Transform of the single row should yield all 0s
    X_scaled = scaler.transform(X)
    np.testing.assert_array_almost_equal(X_scaled, [[0.0, 0.0, 0.0]])


def test_standard_scaler_zero_variance():
    """3. Scaling columns with zero variance (all values identical)."""
    scaler = preprocessing.StandardScaler()
    X = np.array([[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]])
    X_scaled = scaler.fit_transform(X)
    
    # Scale should be set to 1.0 to avoid division by zero, mean should be 5.0, var should be 0.0
    np.testing.assert_array_almost_equal(scaler.mean_, [5.0, 5.0])
    np.testing.assert_array_almost_equal(scaler.var_, [0.0, 0.0])
    np.testing.assert_array_almost_equal(scaler.scale_, [1.0, 1.0])
    np.testing.assert_array_almost_equal(X_scaled, [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])


def test_standard_scaler_extreme_values():
    """4. Extremely large/small numerical inputs to check no overflow/underflow or division by zero."""
    scaler = preprocessing.StandardScaler()
    X = np.array([[1e15, 1e-15], [-1e15, -1e-15]])
    
    scaler.fit(X)
    # The mean should be 0
    np.testing.assert_array_almost_equal(scaler.mean_, [0.0, 0.0])
    # The scale should be valid non-zero
    assert scaler.scale_[0] > 0.0
    assert scaler.scale_[1] > 0.0
    
    X_scaled = scaler.transform(X)
    # Should scale perfectly without NaNs or Infs
    assert np.all(np.isfinite(X_scaled))
    np.testing.assert_array_almost_equal(X_scaled.mean(axis=0), [0.0, 0.0])
    np.testing.assert_array_almost_equal(X_scaled.std(axis=0), [1.0, 1.0])


def test_standard_scaler_invalid_inputs():
    """5. Input containing inf should raise ValueError."""
    scaler = preprocessing.StandardScaler()
    with pytest.raises(ValueError):
        scaler.fit([[1.0, np.inf], [2.0, 3.0]])


# =====================================================================
# MinMaxScaler Boundary Cases
# =====================================================================

def test_min_max_scaler_empty_input():
    """1. Fitting on empty array should raise ValueError."""
    scaler = preprocessing.MinMaxScaler()
    with pytest.raises(ValueError):
        scaler.fit(np.empty((0, 2)))


def test_min_max_scaler_single_row():
    """2. Fitting on single row sets scale to 1.0 and maps to feature_range[0]."""
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    X = np.array([[5.0, 10.0]])
    scaler.fit(X)
    
    np.testing.assert_array_almost_equal(scaler.data_min_, [5.0, 10.0])
    np.testing.assert_array_almost_equal(scaler.data_max_, [5.0, 10.0])
    np.testing.assert_array_almost_equal(scaler.scale_, [1.0, 1.0])
    
    X_scaled = scaler.transform(X)
    np.testing.assert_array_almost_equal(X_scaled, [[0.0, 0.0]])


def test_min_max_scaler_zero_variance():
    """3. Scaling constant features should map them to feature_range[0]."""
    scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    X = np.array([[4.0, 2.0], [4.0, 3.0], [4.0, 4.0]])
    X_scaled = scaler.fit_transform(X)
    
    # First column is constant, should be mapped to -1.0
    np.testing.assert_array_almost_equal(X_scaled[:, 0], [-1.0, -1.0, -1.0])
    np.testing.assert_array_almost_equal(X_scaled[:, 1], [-1.0, 0.0, 1.0])


def test_min_max_scaler_extreme_values():
    """4. Extremely large/small numerical inputs should scale without overflow/underflow."""
    scaler = preprocessing.MinMaxScaler()
    X = np.array([[1e15, 1e-13], [-1e15, -1e-13]])
    X_scaled = scaler.fit_transform(X)
    
    assert np.all(np.isfinite(X_scaled))
    np.testing.assert_array_almost_equal(X_scaled, [[1.0, 1.0], [0.0, 0.0]])


def test_min_max_scaler_invalid_inputs():
    """5. Input containing inf should raise ValueError."""
    scaler = preprocessing.MinMaxScaler()
    with pytest.raises(ValueError):
        scaler.fit([[1.0, 2.0], [np.inf, 3.0]])


# =====================================================================
# LabelEncoder Boundary Cases
# =====================================================================

@pytest.mark.skip(reason='Not supported in thermite')
def test_label_encoder_empty_input():
    """1. Fitting on empty list or array should result in empty classes_."""
    le = preprocessing.LabelEncoder()
    le.fit([])
    assert len(le.classes_) == 0
    np.testing.assert_array_equal(le.transform([]), np.array([]))


def test_label_encoder_single_class():
    """2. Fitting on single class or all identical values."""
    le = preprocessing.LabelEncoder()
    y = np.array(["only_one", "only_one", "only_one"])
    y_encoded = le.fit_transform(y)
    
    assert list(le.classes_) == ["only_one"]
    np.testing.assert_array_equal(y_encoded, [0, 0, 0])
    
    # Inverse transform
    np.testing.assert_array_equal(le.inverse_transform([0]), ["only_one"])


def test_label_encoder_unseen_value():
    """3. Transforming unseen category should raise ValueError."""
    le = preprocessing.LabelEncoder()
    le.fit(["cat", "dog"])
    with pytest.raises(ValueError):
        le.transform(["bird"])


def test_label_encoder_float_values():
    """4. Fitting and transforming float categories, checking sorting order."""
    le = preprocessing.LabelEncoder()
    y = np.array([2.5, 0.5, 1.5, 0.5])
    y_encoded = le.fit_transform(y)
    
    np.testing.assert_array_equal(le.classes_, [0.5, 1.5, 2.5])
    np.testing.assert_array_equal(y_encoded, [2, 0, 1, 0])


def test_label_encoder_high_cardinality():
    """5. High number of unique classes, checking mapping correctness."""
    le = preprocessing.LabelEncoder()
    y = np.array([f"class_{i}" for i in range(1000)])
    le.fit(y)
    
    assert len(le.classes_) == 1000
    res = le.transform(["class_0", "class_999", "class_500"])
    assert res[0] < res[2] < res[1]


# =====================================================================
# OneHotEncoder Boundary Cases
# =====================================================================

def test_one_hot_encoder_empty_input():
    """1. Fitting on empty array should raise ValueError."""
    ohe = preprocessing.OneHotEncoder(sparse_output=False)
    with pytest.raises(ValueError):
        ohe.fit(np.empty((0, 3)))
    with pytest.raises(ValueError):
        ohe.fit(np.empty((5, 0)))


def test_one_hot_encoder_unseen_value_error():
    """2. Transforming unseen category when handle_unknown='error' raises ValueError."""
    ohe = preprocessing.OneHotEncoder(handle_unknown="error", sparse_output=False)
    ohe.fit([["apple"], ["banana"]])
    with pytest.raises(ValueError):
        ohe.transform([["cherry"]])


def test_one_hot_encoder_unseen_value_ignore():
    """3. Transforming unseen category when handle_unknown='ignore' maps to all 0s."""
    ohe = preprocessing.OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    ohe.fit([["apple"], ["banana"]])
    X_encoded = ohe.transform([["cherry"], ["apple"]])
    
    np.testing.assert_array_equal(X_encoded, [[0.0, 0.0], [1.0, 0.0]])


@pytest.mark.skip(reason='Not supported in thermite')
def test_one_hot_encoder_single_category_drop_first():
    """4. If drop='first' is specified and a feature has only 1 category, it returns 0 columns."""
    ohe = preprocessing.OneHotEncoder(drop="first", sparse_output=False)
    X = np.array([["apple"], ["apple"], ["apple"]])
    X_encoded = ohe.fit_transform(X)
    
    assert X_encoded.shape == (3, 0)


@pytest.mark.skip(reason='Not supported in thermite')
def test_one_hot_encoder_custom_categories():
    """5. Specifying custom categories that may or may not exist in the training data."""
    ohe = preprocessing.OneHotEncoder(categories=[["apple", "banana", "cherry"]], sparse_output=False)
    X = np.array([["banana"], ["apple"]])
    X_encoded = ohe.fit_transform(X)
    
    assert X_encoded.shape == (2, 3)
    np.testing.assert_array_equal(X_encoded, [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
