import pytest
import numpy as np
from tests.conftest import get_module

# Dynamically import thermite/sklearn preprocessing module
preprocessing = get_module("preprocessing")


# StandardScaler tests
def test_standard_scaler_basic():
    """Test standard scaling of 2D array, check mean ~0 and std ~1."""
    scaler = preprocessing.StandardScaler()
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    X_scaled = scaler.fit_transform(X)
    
    assert X_scaled.shape == (3, 2)
    np.testing.assert_array_almost_equal(X_scaled.mean(axis=0), [0.0, 0.0])
    np.testing.assert_array_almost_equal(X_scaled.std(axis=0), [1.0, 1.0])


def test_standard_scaler_no_mean():
    """Test standard scaling with with_mean=False and with_std=True."""
    scaler = preprocessing.StandardScaler(with_mean=False, with_std=True)
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    X_scaled = scaler.fit_transform(X)
    
    assert X_scaled.shape == (3, 2)
    # Mean should not be 0, but variance should be scaled
    assert not np.allclose(X_scaled.mean(axis=0), [0.0, 0.0])
    # The std of scaled features should be adjusted appropriately by the scale of the original features
    assert scaler.mean_ is not None


def test_standard_scaler_no_std():
    """Test standard scaling with with_mean=True and with_std=False."""
    scaler = preprocessing.StandardScaler(with_mean=True, with_std=False)
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    X_scaled = scaler.fit_transform(X)
    
    assert X_scaled.shape == (3, 2)
    np.testing.assert_array_almost_equal(X_scaled.mean(axis=0), [0.0, 0.0])
    # Stds should not be 1.0 (they should match original stds)
    orig_std = X.std(axis=0)
    scaled_std = X_scaled.std(axis=0)
    np.testing.assert_array_almost_equal(orig_std, scaled_std)


def test_standard_scaler_fit_transform_separate():
    """Fit on one array and transform another (e.g. out-of-sample data)."""
    scaler = preprocessing.StandardScaler()
    X_train = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    X_test = np.array([[2.0, 3.0], [4.0, 5.0]])
    
    scaler.fit(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    assert X_test_scaled.shape == (2, 2)
    # Manually check scaling of test data using train stats
    mean = np.array([3.0, 4.0])
    scale = np.array([1.63299316, 1.63299316])
    expected = (X_test - mean) / scale
    np.testing.assert_array_almost_equal(X_test_scaled, expected)


def test_standard_scaler_inverse():
    """Fit on 2D array and check inverse_transform works correctly."""
    scaler = preprocessing.StandardScaler()
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    X_scaled = scaler.fit_transform(X)
    X_inv = scaler.inverse_transform(X_scaled)
    
    np.testing.assert_array_almost_equal(X, X_inv)


# MinMaxScaler tests
def test_min_max_scaler_basic():
    """MinMax scaling of 2D array, check min is 0 and max is 1."""
    scaler = preprocessing.MinMaxScaler()
    X = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]])
    X_scaled = scaler.fit_transform(X)
    
    assert X_scaled.shape == (3, 2)
    np.testing.assert_array_almost_equal(X_scaled.min(axis=0), [0.0, 0.0])
    np.testing.assert_array_almost_equal(X_scaled.max(axis=0), [1.0, 1.0])


def test_min_max_scaler_range():
    """MinMax scaling with custom range (e.g. feature_range=(-1, 1))."""
    scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    X = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]])
    X_scaled = scaler.fit_transform(X)
    
    assert X_scaled.shape == (3, 2)
    np.testing.assert_array_almost_equal(X_scaled.min(axis=0), [-1.0, -1.0])
    np.testing.assert_array_almost_equal(X_scaled.max(axis=0), [1.0, 1.0])


def test_min_max_scaler_fit_transform_separate():
    """Fit on one array and transform another (e.g. test set)."""
    scaler = preprocessing.MinMaxScaler()
    X_train = np.array([[1.0, 10.0], [5.0, 30.0]])
    X_test = np.array([[3.0, 20.0]])
    
    scaler.fit(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    assert X_test_scaled.shape == (1, 2)
    np.testing.assert_array_almost_equal(X_test_scaled, [[0.5, 0.5]])


def test_min_max_scaler_inverse():
    """Inverse transform on custom values within/outside range."""
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    X = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]])
    X_scaled = scaler.fit_transform(X)
    X_inv = scaler.inverse_transform(X_scaled)
    
    np.testing.assert_array_almost_equal(X, X_inv)


def test_min_max_scaler_constant_features():
    """Behavior when all values in a feature are constant (returns feature_range[0] or similar)."""
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    X = np.array([[2.0, 10.0], [2.0, 20.0], [2.0, 30.0]])
    X_scaled = scaler.fit_transform(X)
    
    assert X_scaled.shape == (3, 2)
    # The first column is constant (all 2s), scikit-learn sets its scaled values to feature_range[0] (0.0)
    np.testing.assert_array_almost_equal(X_scaled[:, 0], [0.0, 0.0, 0.0])
    np.testing.assert_array_almost_equal(X_scaled[:, 1], [0.0, 0.5, 1.0])


# LabelEncoder tests
def test_label_encoder_basic():
    """Label encoding of string array, verify classes and integers mapped."""
    le = preprocessing.LabelEncoder()
    y = np.array(["cat", "dog", "cat", "bird"])
    y_encoded = le.fit_transform(y)
    
    assert y_encoded.shape == (4,)
    # Alphabetical order: bird (0), cat (1), dog (2)
    np.testing.assert_array_equal(y_encoded, [1, 2, 1, 0])
    assert list(le.classes_) == ["bird", "cat", "dog"]


def test_label_encoder_numeric():
    """Label encoding of numeric values."""
    le = preprocessing.LabelEncoder()
    y = np.array([10, 30, 20, 10])
    y_encoded = le.fit_transform(y)
    
    assert y_encoded.shape == (4,)
    np.testing.assert_array_equal(y_encoded, [0, 2, 1, 0])


def test_label_encoder_fit_transform_separate():
    """Label encoding with transform on test/new data."""
    le = preprocessing.LabelEncoder()
    y_train = np.array(["paris", "paris", "tokyo", "amsterdam"])
    le.fit(y_train)
    
    y_test = np.array(["tokyo", "amsterdam", "paris"])
    y_test_encoded = le.transform(y_test)
    
    # Classes order: amsterdam (0), paris (1), tokyo (2)
    np.testing.assert_array_equal(y_test_encoded, [2, 0, 1])


def test_label_encoder_inverse():
    """Label encoding inverse_transform."""
    le = preprocessing.LabelEncoder()
    y = np.array(["apple", "banana", "banana", "apple"])
    y_encoded = le.fit_transform(y)
    y_decoded = le.inverse_transform(y_encoded)
    
    np.testing.assert_array_equal(y, y_decoded)


def test_label_encoder_single_element():
    """Transform of a single value / list of length 1, check output."""
    le = preprocessing.LabelEncoder()
    le.fit(["a", "b"])
    res = le.transform(["b"])
    
    assert res.shape == (1,)
    assert res[0] == 1


# OneHotEncoder tests
def test_one_hot_encoder_basic():
    """Standard one-hot encoding on 2D categorical array, check shape and content."""
    ohe = preprocessing.OneHotEncoder(sparse_output=False)
    X = np.array([["apple"], ["banana"], ["apple"]])
    X_encoded = ohe.fit_transform(X)
    
    assert X_encoded.shape == (3, 2)
    # apple (1, 0), banana (0, 1) or similar depending on alphabetical sort: apple (0), banana (1)
    np.testing.assert_array_equal(X_encoded, [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])


def test_one_hot_encoder_ignore_unknown():
    """handle_unknown='ignore' and transforming unseen categories."""
    ohe = preprocessing.OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_train = np.array([["apple"], ["banana"]])
    ohe.fit(X_train)
    
    X_test = np.array([["banana"], ["cherry"], ["apple"]])
    X_encoded = ohe.transform(X_test)
    
    assert X_encoded.shape == (3, 2)
    # cherry is unknown, should map to all 0s
    np.testing.assert_array_equal(X_encoded, [[0.0, 1.0], [0.0, 0.0], [1.0, 0.0]])


@pytest.mark.skip(reason='Not supported in thermite')
def test_one_hot_encoder_drop_first():
    """drop='first' option, check shape and drop behavior."""
    ohe = preprocessing.OneHotEncoder(drop="first", sparse_output=False)
    X = np.array([["apple"], ["banana"], ["cherry"]])
    X_encoded = ohe.fit_transform(X)
    
    # Categories: apple, banana, cherry. apple is dropped, so we have banana, cherry
    assert X_encoded.shape == (3, 2)
    np.testing.assert_array_equal(X_encoded, [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])


@pytest.mark.skip(reason='Not supported in thermite')
def test_one_hot_encoder_custom_categories():
    """Passing custom categories list, check result."""
    ohe = preprocessing.OneHotEncoder(categories=[["apple", "banana", "cherry"]], sparse_output=False)
    X = np.array([["banana"], ["apple"]])
    X_encoded = ohe.fit_transform(X)
    
    assert X_encoded.shape == (2, 3)
    np.testing.assert_array_equal(X_encoded, [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])


def test_one_hot_encoder_inverse_transform():
    """inverse_transform on the one-hot encoded matrix."""
    ohe = preprocessing.OneHotEncoder(sparse_output=False)
    X = np.array([["apple"], ["banana"], ["apple"]])
    X_encoded = ohe.fit_transform(X)
    X_decoded = ohe.inverse_transform(X_encoded)
    
    np.testing.assert_array_equal(X, X_decoded)
