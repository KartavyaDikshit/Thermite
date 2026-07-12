import numpy as np
import thermite.preprocessing as th_pre
import thermite.model_selection as th_mod
import sklearn.preprocessing as sk_pre
import sklearn.model_selection as sk_mod
import sys

def check_scalers_nan_inf():
    print("--- 1. Testing Scalers with NaN / Inf ---")
    scalers_to_test = [
        ("StandardScaler", th_pre.StandardScaler, sk_pre.StandardScaler),
        ("MinMaxScaler", th_pre.MinMaxScaler, sk_pre.MinMaxScaler)
    ]
    
    X_nan = np.array([[1.0, 2.0], [np.nan, 4.0]])
    X_inf = np.array([[1.0, 2.0], [np.inf, 4.0]])
    X_good = np.array([[1.0, 2.0], [3.0, 4.0]])
    
    for name, ThScaler, SkScaler in scalers_to_test:
        # Fit with NaN
        try:
            ThScaler().fit(X_nan)
            print(f"[FAIL] {name}.fit did not raise ValueError on NaN")
        except ValueError as e:
            print(f"[PASS] {name}.fit raised ValueError on NaN: {e}")
            
        # Fit with Inf
        try:
            ThScaler().fit(X_inf)
            print(f"[FAIL] {name}.fit did not raise ValueError on Inf")
        except ValueError as e:
            print(f"[PASS] {name}.fit raised ValueError on Inf: {e}")
            
        # Transform with NaN
        scaler = ThScaler().fit(X_good)
        try:
            scaler.transform(X_nan)
            print(f"[FAIL] {name}.transform did not raise ValueError on NaN")
        except ValueError as e:
            print(f"[PASS] {name}.transform raised ValueError on NaN: {e}")
            
        # Transform with Inf
        try:
            scaler.transform(X_inf)
            print(f"[FAIL] {name}.transform did not raise ValueError on Inf")
        except ValueError as e:
            print(f"[PASS] {name}.transform raised ValueError on Inf: {e}")
            
        # Inverse transform with NaN
        try:
            scaler.inverse_transform(X_nan)
            print(f"[FAIL] {name}.inverse_transform did not raise ValueError on NaN")
        except ValueError as e:
            print(f"[PASS] {name}.inverse_transform raised ValueError on NaN: {e}")

def check_empty_arrays():
    print("\n--- 2. Testing Empty Arrays and Matrices ---")
    
    # 2.1 StandardScaler and MinMaxScaler
    for name, ThScaler in [("StandardScaler", th_pre.StandardScaler), ("MinMaxScaler", th_pre.MinMaxScaler)]:
        # empty fit
        try:
            ThScaler().fit(np.empty((0, 5)))
            print(f"[FAIL] {name}.fit accepted shape (0, 5)")
        except ValueError as e:
            print(f"[PASS] {name}.fit rejected shape (0, 5): {e}")
            
        try:
            ThScaler().fit(np.empty((5, 0)))
            print(f"[FAIL] {name}.fit accepted shape (5, 0)")
        except ValueError as e:
            print(f"[PASS] {name}.fit rejected shape (5, 0): {e}")
            
        # empty transform / inverse_transform (after good fit)
        scaler = ThScaler().fit(np.random.randn(5, 2))
        
        # sklearn raises ValueError on empty transform, what does thermite do?
        try:
            res = scaler.transform(np.empty((0, 2)))
            print(f"[DIVERGENCE] {name}.transform accepted shape (0, 2), returned shape {res.shape} (sklearn raises ValueError)")
        except ValueError as e:
            print(f"[PASS] {name}.transform rejected shape (0, 2): {e}")
            
        try:
            res = scaler.inverse_transform(np.empty((0, 2)))
            print(f"[DIVERGENCE] {name}.inverse_transform accepted shape (0, 2), returned shape {res.shape} (sklearn raises ValueError)")
        except ValueError as e:
            print(f"[PASS] {name}.inverse_transform rejected shape (0, 2): {e}")
            
    # 2.2 LabelEncoder
    le = th_pre.LabelEncoder()
    try:
        le.fit([])
        print("[PASS] LabelEncoder.fit accepted empty 1D list")
        res = le.transform([])
        print(f"[PASS] LabelEncoder.transform returned empty 1D array: {res}")
        res_inv = le.inverse_transform([])
        print(f"[PASS] LabelEncoder.inverse_transform returned empty 1D array: {res_inv}")
    except Exception as e:
        print(f"[FAIL] LabelEncoder failed on empty input: {type(e).__name__}: {e}")
        
    # 2.3 OneHotEncoder
    ohe = th_pre.OneHotEncoder()
    try:
        ohe.fit(np.empty((0, 3)))
        # sklearn raises ValueError on empty fit for OneHotEncoder
        print("[DIVERGENCE] OneHotEncoder.fit accepted shape (0, 3) (sklearn raises ValueError)")
    except ValueError as e:
        print(f"[PASS] OneHotEncoder.fit rejected shape (0, 3): {e}")
        
    # 2.4 train_test_split
    try:
        th_mod.train_test_split(np.empty((0, 5)))
        print("[FAIL] train_test_split accepted empty array")
    except ValueError as e:
        print(f"[PASS] train_test_split rejected empty array: {e}")

def check_stratification_imbalanced():
    print("\n--- 3. Testing Stratification with Imbalanced Classes ---")
    
    # 3.1 Class with 1 sample
    X = np.random.randn(10, 2)
    y = np.array([0]*9 + [1])
    try:
        # sklearn raises ValueError because class '1' has only 1 sample
        X_tr, X_te, y_tr, y_te = th_mod.train_test_split(X, y, test_size=0.2, stratify=y)
        print(f"[DIVERGENCE] train_test_split with stratify accepted class of size 1 (train/test split: {len(y_tr)}/{len(y_te)})")
        print(f"            y_train classes count: {np.bincount(y_tr)}, y_test classes count: {np.bincount(y_te)}")
    except ValueError as e:
        print(f"[PASS] train_test_split rejected class of size 1: {e}")
        
    # 3.2 Extremely imbalanced classes
    X2 = np.random.randn(100, 2)
    y2 = np.array([0]*98 + [1]*2)
    try:
        # test_size=0.2 means test has 20 samples, train has 80 samples.
        # Ratio of class 1 is 2%. Expected: 2% of 20 = 0.4 -> 0 or 1 class 1 samples in test.
        X_tr, X_te, y_tr, y_te = th_mod.train_test_split(X2, y2, test_size=0.2, stratify=y2)
        print(f"[PASS] train_test_split succeeded with class ratio 98:2")
        print(f"            y_train classes count: {np.bincount(y_tr)}, y_test classes count: {np.bincount(y_te)}")
    except Exception as e:
        print(f"[FAIL] train_test_split failed with class ratio 98:2: {e}")

def check_shuffling_reproducibility():
    print("\n--- 4. Testing Shuffling Randomness and Reproducibility ---")
    
    X = np.arange(100).reshape(50, 2)
    y = np.arange(50)
    
    # 4.1 Reproducibility of same seed
    X_tr1, X_te1, y_tr1, y_te1 = th_mod.train_test_split(X, y, test_size=0.2, random_state=42)
    X_tr2, X_te2, y_tr2, y_te2 = th_mod.train_test_split(X, y, test_size=0.2, random_state=42)
    
    if np.array_equal(X_tr1, X_tr2) and np.array_equal(X_te1, X_te2):
        print("[PASS] Same seed (42) produces identical splits")
    else:
        print("[FAIL] Same seed (42) produces different splits")
        
    # 4.2 Difference with different seed
    X_tr3, X_te3, y_tr3, y_te3 = th_mod.train_test_split(X, y, test_size=0.2, random_state=43)
    
    if not np.array_equal(X_tr1, X_tr3):
        print("[PASS] Different seed (43) produces different splits")
    else:
        print("[FAIL] Different seed (43) produced identical splits to seed 42")
        
    # 4.3 Shuffle = False behavior
    X_tr_ns, X_te_ns, y_tr_ns, y_te_ns = th_mod.train_test_split(X, y, test_size=0.2, shuffle=False)
    # Sklearn puts train first, test last: train = X[:40], test = X[40:]
    # Let's check what thermite did:
    if np.array_equal(y_tr_ns, np.arange(10, 50)) and np.array_equal(y_te_ns, np.arange(10)):
        print("[DIVERGENCE] train_test_split(shuffle=False) puts test first and train last (opposite of sklearn)")
    elif np.array_equal(y_tr_ns, np.arange(40)) and np.array_equal(y_te_ns, np.arange(40, 50)):
        print("[PASS] train_test_split(shuffle=False) matches sklearn behavior (train first, test last)")
    else:
        print(f"[INFO] train_test_split(shuffle=False) split at: train={y_tr_ns}, test={y_te_ns}")

def check_unknown_categories():
    print("\n--- 5. Testing Unknown Categories in Encoders ---")
    
    # 5.1 OneHotEncoder with handle_unknown='error'
    ohe_err = th_pre.OneHotEncoder(handle_unknown='error')
    ohe_err.fit([['a'], ['b']])
    try:
        ohe_err.transform([['c']])
        print("[FAIL] OneHotEncoder(handle_unknown='error') did not raise ValueError on unknown category")
    except ValueError as e:
        print(f"[PASS] OneHotEncoder(handle_unknown='error') raised ValueError: {e}")
        
    # 5.2 OneHotEncoder with handle_unknown='ignore'
    ohe_ign = th_pre.OneHotEncoder(handle_unknown='ignore')
    ohe_ign.fit([['a'], ['b']])
    try:
        res = ohe_ign.transform([['c']])
        expected = np.array([[0.0, 0.0]])
        if np.array_equal(res, expected):
            print(f"[PASS] OneHotEncoder(handle_unknown='ignore') maps unknown category to all zeros: {res}")
        else:
            print(f"[FAIL] OneHotEncoder(handle_unknown='ignore') mapped unknown category to: {res}")
    except Exception as e:
        print(f"[FAIL] OneHotEncoder(handle_unknown='ignore') failed on transform: {e}")

if __name__ == "__main__":
    check_scalers_nan_inf()
    check_empty_arrays()
    check_stratification_imbalanced()
    check_shuffling_reproducibility()
    check_unknown_categories()
