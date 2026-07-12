import numpy as np
import sklearn.preprocessing as sk_pre
import sklearn.model_selection as sk_mod
import thermite.preprocessing as th_pre
import thermite.model_selection as th_mod

# 1. train_test_split test (with balanced classes for exact ratio match)
X = np.random.randn(100, 4)
y = np.array([0] * 50 + [1] * 50)
X_tr, X_te, y_tr, y_te = th_mod.train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
assert X_tr.shape == (80, 4)
assert X_te.shape == (20, 4)
# Check stratify ratios
assert np.sum(y_tr == 1) / len(y_tr) == np.sum(y_te == 1) / len(y_te)

# 2. StandardScaler test
X_scale = np.random.randn(200, 5)
sk_scaler = sk_pre.StandardScaler().fit(X_scale)
th_scaler = th_pre.StandardScaler().fit(X_scale)
assert np.allclose(sk_scaler.mean_, th_scaler.mean_)
assert np.allclose(sk_scaler.var_, th_scaler.var_)
assert np.allclose(sk_scaler.transform(X_scale), th_scaler.transform(X_scale))
assert np.allclose(X_scale, th_scaler.inverse_transform(th_scaler.transform(X_scale)))

# 3. MinMaxScaler test
sk_minmax = sk_pre.MinMaxScaler(feature_range=(-1, 1)).fit(X_scale)
th_minmax = th_pre.MinMaxScaler(feature_range=(-1, 1)).fit(X_scale)
assert np.allclose(sk_minmax.data_min_, th_minmax.data_min_)
assert np.allclose(sk_minmax.data_max_, th_minmax.data_max_)
assert np.allclose(sk_minmax.transform(X_scale), th_minmax.transform(X_scale))

# 4. LabelEncoder test
y_labels = np.array(["cat", "dog", "cat", "bird", "dog"])
sk_le = sk_pre.LabelEncoder().fit(y_labels)
th_le = th_pre.LabelEncoder().fit(y_labels)
assert np.array_equal(sk_le.classes_, th_le.classes_)
assert np.array_equal(sk_le.transform(y_labels), th_le.transform(y_labels))

# 5. OneHotEncoder test
X_cat = np.array([["int_1", "str_A"], ["int_2", "str_B"], ["int_1", "str_A"]], dtype=object)
sk_ohe = sk_pre.OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(X_cat)
th_ohe = th_pre.OneHotEncoder(handle_unknown='ignore').fit(X_cat)
assert np.allclose(sk_ohe.transform(X_cat), th_ohe.transform(X_cat))
print("All Python verification tests pass!")
