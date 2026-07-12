import numpy as np
import traceback
import sys

import thermite.preprocessing as th_pre
import thermite.model_selection as th_mod

def run_test(name, func):
    print(f"--- Running {name} ---")
    try:
        func()
        print(f"[PASS] {name}\n")
        return True
    except Exception as e:
        print(f"[FAIL] {name}")
        traceback.print_exc()
        print()
        return False

# =====================================================================
# 1. Infinite / NaN values in scaler inputs (should raise ValueError)
# =====================================================================
def test_scaler_nan_inf():
    # Test StandardScaler fit with NaN
    scaler = th_pre.StandardScaler()
    X_nan = np.array([[1.0, 2.0], [3.0, np.nan]])
    try:
        scaler.fit(X_nan)
        raise AssertionError("StandardScaler fit accepted NaN without raising ValueError")
    except ValueError as e:
        print(f"StandardScaler fit NaN (expected exception): {e}")

    # Test StandardScaler fit with Inf
    X_inf = np.array([[1.0, 2.0], [3.0, np.inf]])
    try:
        scaler.fit(X_inf)
        raise AssertionError("StandardScaler fit accepted Inf without raising ValueError")
    except ValueError as e:
        print(f"StandardScaler fit Inf (expected exception): {e}")

    # Test StandardScaler fit with -Inf
    X_neginf = np.array([[1.0, 2.0], [3.0, -np.inf]])
    try:
        scaler.fit(X_neginf)
        raise AssertionError("StandardScaler fit accepted -Inf without raising ValueError")
    except ValueError as e:
        print(f"StandardScaler fit -Inf (expected exception): {e}")

    # Test StandardScaler transform with NaN/Inf after fitting with clean data
    X_clean = np.array([[1.0, 2.0], [3.0, 4.0]])
    scaler.fit(X_clean)
    
    try:
        scaler.transform(X_nan)
        raise AssertionError("StandardScaler transform accepted NaN without raising ValueError")
    except ValueError as e:
        print(f"StandardScaler transform NaN (expected exception): {e}")

    try:
        scaler.transform(X_inf)
        raise AssertionError("StandardScaler transform accepted Inf without raising ValueError")
    except ValueError as e:
        print(f"StandardScaler transform Inf (expected exception): {e}")

    # Test MinMaxScaler fit with NaN
    scaler_mm = th_pre.MinMaxScaler()
    try:
        scaler_mm.fit(X_nan)
        raise AssertionError("MinMaxScaler fit accepted NaN without raising ValueError")
    except ValueError as e:
        print(f"MinMaxScaler fit NaN (expected exception): {e}")

    # Test MinMaxScaler fit with Inf
    try:
        scaler_mm.fit(X_inf)
        raise AssertionError("MinMaxScaler fit accepted Inf without raising ValueError")
    except ValueError as e:
        print(f"MinMaxScaler fit Inf (expected exception): {e}")

    # Test MinMaxScaler transform with NaN/Inf after fitting with clean data
    scaler_mm.fit(X_clean)
    try:
        scaler_mm.transform(X_nan)
        raise AssertionError("MinMaxScaler transform accepted NaN without raising ValueError")
    except ValueError as e:
        print(f"MinMaxScaler transform NaN (expected exception): {e}")

# =====================================================================
# 2. Empty arrays or matrices
# =====================================================================
def test_empty_inputs():
    # StandardScaler
    scaler = th_pre.StandardScaler()
    for shape in [(0, 2), (2, 0), (0, 0)]:
        try:
            scaler.fit(np.empty(shape))
            raise AssertionError(f"StandardScaler fit accepted empty array of shape {shape}")
        except ValueError as e:
            print(f"StandardScaler fit empty shape {shape} (expected exception): {e}")

    # MinMaxScaler
    scaler_mm = th_pre.MinMaxScaler()
    for shape in [(0, 2), (2, 0), (0, 0)]:
        try:
            scaler_mm.fit(np.empty(shape))
            raise AssertionError(f"MinMaxScaler fit accepted empty array of shape {shape}")
        except ValueError as e:
            print(f"MinMaxScaler fit empty shape {shape} (expected exception): {e}")

    # LabelEncoder
    le = th_pre.LabelEncoder()
    try:
        le.fit(np.empty((0,)))
        # Let's see if fit succeeds. If it does, does classes_ look empty?
        print(f"LabelEncoder fit empty array succeeded. classes_: {le.classes_}")
        # Let's try transform on empty
        trans = le.transform(np.empty((0,)))
        print(f"LabelEncoder transform empty array succeeded: {trans}")
    except Exception as e:
        print(f"LabelEncoder fit/transform empty array raised exception: {e}")

    # OneHotEncoder
    ohe = th_pre.OneHotEncoder()
    for shape in [(0, 2), (2, 0), (0, 0)]:
        try:
            # Fit on empty
            ohe.fit(np.empty(shape))
            print(f"OneHotEncoder fit empty shape {shape} succeeded. categories_: {ohe.categories_}")
            trans = ohe.transform(np.empty(shape))
            print(f"OneHotEncoder transform empty shape {shape} succeeded. Result shape: {trans.shape}")
        except Exception as e:
            print(f"OneHotEncoder fit/transform empty shape {shape} raised exception: {e}")

    # train_test_split
    try:
        th_mod.train_test_split(np.empty((0, 5)))
        raise AssertionError("train_test_split accepted empty array")
    except ValueError as e:
        print(f"train_test_split empty array (expected exception): {e}")

# =====================================================================
# 3. Stratification with extremely small or imbalanced classes
# =====================================================================
def test_stratification_imbalanced():
    # Test class with 1 member
    X = np.random.randn(10, 2)
    y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]) # Class 1 has only 1 member
    
    print("Testing train_test_split stratify with a class of size 1...")
    try:
        X_train, X_test, y_train, y_test = th_mod.train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        print(f"Split succeeded! Train size: {len(y_train)}, Test size: {len(y_test)}")
        print(f"y_train: {y_train}")
        print(f"y_test: {y_test}")
        # Note: Scikit-learn raises ValueError: The least populated class in y has only 1 member...
    except Exception as e:
        print(f"Split failed (as scikit-learn does): {e}")

    # Test highly imbalanced classes
    y_imbalanced = np.array([0] * 98 + [1] * 2) # Size 100, class 1 has 2 members
    X_imb = np.random.randn(100, 2)
    try:
        X_train, X_test, y_train, y_test = th_mod.train_test_split(
            X_imb, y_imbalanced, test_size=0.1, random_state=42, stratify=y_imbalanced
        )
        print(f"Highly imbalanced split succeeded! Train size: {len(y_train)}, Test size: {len(y_test)}")
        print(f"y_train counts: 0: {np.sum(y_train == 0)}, 1: {np.sum(y_train == 1)}")
        print(f"y_test counts: 0: {np.sum(y_test == 0)}, 1: {np.sum(y_test == 1)}")
    except Exception as e:
        print(f"Highly imbalanced split failed: {e}")

# =====================================================================
# 4. Shuffling randomness reproducibility
# =====================================================================
def test_shuffling_reproducibility():
    X = np.arange(100).reshape((50, 2))
    y = np.arange(50)

    # 1. Verify same seed produces identical splits
    X_tr1, X_te1, y_tr1, y_te1 = th_mod.train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    X_tr2, X_te2, y_tr2, y_te2 = th_mod.train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    
    assert np.array_equal(X_tr1, X_tr2), "Same seed produced different train splits for X"
    assert np.array_equal(X_te1, X_te2), "Same seed produced different test splits for X"
    assert np.array_equal(y_tr1, y_tr2), "Same seed produced different train splits for y"
    assert np.array_equal(y_te1, y_te2), "Same seed produced different test splits for y"
    print("Same seed produces identical splits: Verified.")

    # 2. Verify different seed produces different splits
    X_tr3, X_te3, y_tr3, y_te3 = th_mod.train_test_split(X, y, test_size=0.2, random_state=43, shuffle=True)
    
    # It is statistically possible but highly unlikely to get identical splits with different seeds
    assert not np.array_equal(X_tr1, X_tr3), "Different seeds produced identical train splits"
    print("Different seeds produce different splits: Verified.")

    # 3. Verify shuffle=False is deterministic and independent of random_state
    X_tr_ns1, X_te_ns1 = th_mod.train_test_split(X, test_size=0.2, random_state=42, shuffle=False)
    X_tr_ns2, X_te_ns2 = th_mod.train_test_split(X, test_size=0.2, random_state=100, shuffle=False)
    
    assert np.array_equal(X_tr_ns1, X_tr_ns2), "shuffle=False was affected by random_state"
    print("shuffle=False is deterministic and independent of seed: Verified.")

# =====================================================================
# 5. Unknown categories in encoders (handle_unknown)
# =====================================================================
def test_unknown_categories():
    # LabelEncoder unknown categories
    le = th_pre.LabelEncoder()
    le.fit([1, 2, 3])
    try:
        le.transform([1, 4])
        raise AssertionError("LabelEncoder transform accepted unknown category without raising ValueError")
    except ValueError as e:
        print(f"LabelEncoder transform unknown category (expected exception): {e}")

    # OneHotEncoder handle_unknown='error'
    ohe_err = th_pre.OneHotEncoder(handle_unknown='error')
    ohe_err.fit([['apple'], ['banana']])
    try:
        ohe_err.transform([['cherry']])
        raise AssertionError("OneHotEncoder transform with handle_unknown='error' accepted unknown category without raising ValueError")
    except ValueError as e:
        print(f"OneHotEncoder (handle_unknown='error') unknown category (expected exception): {e}")

    # OneHotEncoder handle_unknown='ignore'
    ohe_ign = th_pre.OneHotEncoder(handle_unknown='ignore')
    ohe_ign.fit([['apple'], ['banana']])
    try:
        res = ohe_ign.transform([['apple'], ['cherry'], ['banana']])
        print(f"OneHotEncoder (handle_unknown='ignore') transform result:\n{res}")
        # Expected: apple -> [1, 0], cherry -> [0, 0], banana -> [0, 1]
        expected = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
        assert np.array_equal(res, expected), f"Expected:\n{expected}\nGot:\n{res}"
        print("OneHotEncoder handle_unknown='ignore' verified.")
    except Exception as e:
        print(f"OneHotEncoder (handle_unknown='ignore') failed: {e}")

if __name__ == "__main__":
    tests = {
        "Scaler NaN/Inf Inputs": test_scaler_nan_inf,
        "Empty Inputs": test_empty_inputs,
        "Stratification and Imbalanced Classes": test_stratification_imbalanced,
        "Shuffling Randomness Reproducibility": test_shuffling_reproducibility,
        "Unknown Categories in Encoders": test_unknown_categories
    }
    
    success = True
    for name, func in tests.items():
        if not run_test(name, func):
            success = False
            
    if not success:
        print("Some tests failed!")
        sys.exit(1)
    else:
        print("All challenge tests executed successfully!")
        sys.exit(0)
