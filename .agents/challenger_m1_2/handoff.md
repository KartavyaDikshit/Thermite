# Handoff Report: Milestone 1 Verification and Robustness Testing

## 1. Observation

### Execution of `challenge_m1_2.py`
Running the custom script `.agents/challenger_m1_2/challenge_m1_2.py` outputted:
```
--- 1. Testing Scalers with NaN / Inf ---
[PASS] StandardScaler.fit raised ValueError on NaN: Input contains NaN or infinity values
[PASS] StandardScaler.fit raised ValueError on Inf: Input contains NaN or infinity values
[PASS] StandardScaler.transform raised ValueError on NaN: Input contains NaN or infinity values
[PASS] StandardScaler.transform raised ValueError on Inf: Input contains NaN or infinity values
[PASS] StandardScaler.inverse_transform raised ValueError on NaN: Input contains NaN or infinity values
[PASS] MinMaxScaler.fit raised ValueError on NaN: Input contains NaN or infinity values
[PASS] MinMaxScaler.fit raised ValueError on Inf: Input contains NaN or infinity values
[PASS] MinMaxScaler.transform raised ValueError on NaN: Input contains NaN or infinity values
[PASS] MinMaxScaler.transform raised ValueError on Inf: Input contains NaN or infinity values
[PASS] MinMaxScaler.inverse_transform raised ValueError on NaN: Input contains NaN or infinity values

--- 2. Testing Empty Arrays and Matrices ---
[PASS] StandardScaler.fit rejected shape (0, 5): Input array is empty
[PASS] StandardScaler.fit rejected shape (5, 0): Input array is empty
[DIVERGENCE] StandardScaler.transform accepted shape (0, 2), returned shape (0, 2) (sklearn raises ValueError)
[DIVERGENCE] StandardScaler.inverse_transform accepted shape (0, 2), returned shape (0, 2) (sklearn raises ValueError)
[PASS] MinMaxScaler.fit rejected shape (0, 5): Input array is empty
[PASS] MinMaxScaler.fit rejected shape (5, 0): Input array is empty
[DIVERGENCE] MinMaxScaler.transform accepted shape (0, 2), returned shape (0, 2) (sklearn raises ValueError)
[DIVERGENCE] MinMaxScaler.inverse_transform accepted shape (0, 2), returned shape (0, 2) (sklearn raises ValueError)
[PASS] LabelEncoder.fit accepted empty 1D list
[PASS] LabelEncoder.transform returned empty 1D array: []
[PASS] LabelEncoder.inverse_transform returned empty 1D array: []
[DIVERGENCE] OneHotEncoder.fit accepted shape (0, 3) (sklearn raises ValueError)
[PASS] train_test_split rejected empty array: Number of samples must be greater than 0

--- 3. Testing Stratification with Imbalanced Classes ---
[DIVERGENCE] train_test_split with stratify accepted class of size 1 (train/test split: 8/2)
            y_train classes count: [7 1], y_test classes count: [2]
[PASS] train_test_split succeeded with class ratio 98:2
            y_train classes count: [78  2], y_test classes count: [20]

--- 4. Testing Shuffling Randomness and Reproducibility ---
[PASS] Same seed (42) produces identical splits
[PASS] Different seed (43) produces different splits
[DIVERGENCE] train_test_split(shuffle=False) puts test first and train last (opposite of sklearn)

--- 5. Testing Unknown Categories in Encoders ---
[PASS] OneHotEncoder(handle_unknown='error') raised ValueError: Found unknown category c in column 0
[PASS] OneHotEncoder(handle_unknown='ignore') maps unknown category to all zeros: [[0. 0.]]
```

### Preprocessing Test Failures in Existing `pytest`
Running `PYTHONPATH=. .venv_homebrew/bin/pytest tests/test_tier1_preprocessing.py tests/test_tier2_preprocessing.py` yielded 13 failures. Key observations:
1. `OneHotEncoder.__init__() got an unexpected keyword argument 'sparse_output'` (and other arguments like `drop`, `categories`):
   ```python
   ohe = preprocessing.OneHotEncoder(sparse_output=False)
   TypeError: OneHotEncoder.__init__() got an unexpected keyword argument 'sparse_output'
   ```
2. `MinMaxScaler` single row test failure:
   ```
   np.testing.assert_array_almost_equal(scaler.scale_, [1.0, 1.0])
   AssertionError: 
   ACTUAL: array([0., 0.])
   DESIRED: array([1., 1.])
   ```
3. `LabelEncoder` float values test failure:
   ```
   np.testing.assert_array_equal(le.classes_, [0.5, 1.5, 2.5])
   AssertionError: 
   ACTUAL: array(['0.5', '1.5', '2.5'], dtype='<U3')
   DESIRED: array([0.5, 1.5, 2.5])
   ```

---

## 2. Logic Chain

1. **StandardScaler & MinMaxScaler (Empty Arrays)**:
   - *Observation*: `StandardScaler.transform` and `MinMaxScaler.transform` accept empty arrays `(0, 2)` and successfully return empty arrays.
   - *Comparison*: `sklearn.preprocessing.StandardScaler().transform` validates the input using `check_array`, which rejects arrays with 0 samples by default (raising a `ValueError`).
   - *Inference*: Thermite's Rust code in `crates/thermite-core/src/preprocessing.rs` does not check for empty arrays inside `transform` or `inverse_transform` methods, leading to this divergence.

2. **OneHotEncoder (Empty Fit)**:
   - *Observation*: `OneHotEncoder().fit(np.empty((0, 3)))` succeeds in Thermite.
   - *Comparison*: `sklearn.preprocessing.OneHotEncoder().fit` raises `ValueError: Found array with 0 sample(s)...`.
   - *Inference*: `OneHotEncoder`'s wrapper and Rust implementation lack empty-input validation checks in the `fit` method.

3. **Stratification with Single-Sample Classes**:
   - *Observation*: `train_test_split` with `stratify=y` succeeds when `y` contains a class with only 1 member (e.g. `y = [0]*9 + [1]`), allocating it only to the training set.
   - *Comparison*: `sklearn.model_selection.train_test_split` raises a `ValueError` because a class cannot be stratified with only 1 sample (minimum required is 2).
   - *Inference*: Thermite's `compute_split_indices` in Rust handles remainder distribution using Largest Remainder Method but does not check if any class contains fewer than 2 samples before attempting stratification.

4. **Shuffle=False Split Order**:
   - *Observation*: `train_test_split(..., shuffle=False)` puts the test set first (`0..n_test`) and the train set last, yielding `y_test = [0, 1, 2]` and `y_train = [3, 4, 5, 6, 7, 8, 9]`.
   - *Comparison*: `sklearn`'s non-shuffled split allocates the train set first and test set last (e.g. `y_train = [0, 1, 2, 3, 4, 5, 6]` and `y_test = [7, 8, 9]`).
   - *Inference*: Thermite's Rust implementation `indices.split_at(n_test)` splits the test set from the front, reversing the standard convention.

5. **MinMaxScaler scale_ for Zero Difference**:
   - *Observation*: Fitting `MinMaxScaler` on a single row or constant features sets `scale_` to `0.0`.
   - *Comparison*: `sklearn` sets `scale_` to `1.0` in this case.
   - *Inference*: In Rust, `if diff == 0.0 { scale_arr[i] = 0.0; }` handles constant features by outputting zero instead of unity.

6. **LabelEncoder Type Casting**:
   - *Observation*: Fitting `LabelEncoder` on a float array `[2.5, 0.5, 1.5]` produces classes of type string `['0.5', '1.5', '2.5']`.
   - *Comparison*: `sklearn` preserves the float data type.
   - *Inference*: Thermite's `LabelEncoder` implementation in `thermite/preprocessing.py` checks `np.issubdtype(y.dtype, np.integer)`. If false, it falls back to casting all elements to strings before fitting.

---

## 3. Caveats
- No testing was performed on other upcoming packages (linear model, clustering, etc.) as they are out of scope for Milestone 1.
- Randomness testing was performed on a single platform (macOS, zsh), but since `rand::rngs::SmallRng` is deterministic given a seed, platform discrepancies are unlikely to affect reproducibility.

---

## 4. Conclusion
There are multiple functional divergences and bug locations in the Milestone 1 preprocessing and model selection classes.
- **Risk Level**: HIGH.
  - The type casting issues in `LabelEncoder` and parameter mismatches in `OneHotEncoder` cause E2E test suites to fail.
  - The `shuffle=False` order in `train_test_split` is opposite to `scikit-learn` convention.
  - Empty array and single-sample stratification checks are missing.

---

## 5. Verification Method

To verify these results independently:
1. Run the custom test script:
   ```bash
   PYTHONPATH=. .venv_homebrew/bin/python .agents/challenger_m1_2/challenge_m1_2.py
   ```
2. Inspect the test suite run results:
   ```bash
   PYTHONPATH=. .venv_homebrew/bin/pytest tests/test_tier1_preprocessing.py tests/test_tier2_preprocessing.py
   ```

---

## 6. Adversarial Review Report

### Challenge Summary
- **Overall risk assessment**: HIGH

### Challenges

#### [High] Challenge 1: train_test_split(shuffle=False) Split Convention
- **Assumption challenged**: The test split is expected to be placed after the train split.
- **Attack scenario**: Code migrating from `sklearn` to `thermite` assumes `X_train, X_test = train_test_split(X, shuffle=False)` maintains chronological order (e.g. time-series). In Thermite, the test indices are from the beginning of the time-series instead of the end.
- **Blast radius**: High. Models trained on historical data will accidentally train on future data (data leakage/lookahead bias).
- **Mitigation**: Change `model_selection.rs` to split train set first: `let (train_part, test_part) = indices.split_at(n_train);`.

#### [High] Challenge 2: LabelEncoder Float/Numeric Casting
- **Assumption challenged**: `LabelEncoder` can fit numeric classes.
- **Attack scenario**: Fitting `LabelEncoder` on a float target variable causes the encoder to store classes as strings. `inverse_transform` will return string objects rather than float primitives, breaking downstream typing.
- **Blast radius**: Medium-High. Code expecting numeric data types will fail downstream with `TypeError`.
- **Mitigation**: Support native float array processing or general object/string conversion without permanent string-only coercion.

#### [Medium] Challenge 3: MinMaxScaler scale_ on Zero Range
- **Assumption challenged**: A constant column retains scale information as 1.0 (unity).
- **Attack scenario**: Feature scaling on a single row or columns with uniform values assigns a scale multiplier of `0.0`. Under `inverse_transform`, division by `scale_` will return NaNs or fail.
- **Blast radius**: Medium.
- **Mitigation**: Update Rust code in `preprocessing.rs` to set `scale_arr[i] = 1.0` when `diff == 0.0`.

#### [Medium] Challenge 4: Missing Input Constraints (NaN/Inf vs Empty)
- **Assumption challenged**: Empty inputs are checked comprehensively.
- **Attack scenario**: Calling `transform` or `inverse_transform` on empty inputs does not raise `ValueError` as in scikit-learn.
- **Blast radius**: Low-Medium. Less standard usage, but diverges from scikit-learn.
- **Mitigation**: Add checks in python/Rust to validate `X.shape[0] > 0` and `X.shape[1] > 0` before processing.

### Stress Test Results
- **NaN / Inf in fit/transform** → Raise `ValueError` → Raised `ValueError` → **PASS**
- **Empty input in MinMaxScaler/StandardScaler fit** → Raise `ValueError` → Raised `ValueError` → **PASS**
- **Empty input in MinMaxScaler/StandardScaler transform** → Raise `ValueError` → Returned empty array (no error) → **FAIL (DIVERGENCE)**
- **OneHotEncoder empty fit** → Raise `ValueError` → Succeeded (no error) → **FAIL (DIVERGENCE)**
- **train_test_split stratify with 1 sample** → Raise `ValueError` → Succeeded (no error) → **FAIL (DIVERGENCE)**
- **train_test_split shuffle=False split order** → Train first, test last → Test first, train last → **FAIL (DIVERGENCE)**

### Unchallenged Areas
- Rayon parallel performance scaling was not stress-tested under high CPU core counts due to single-machine execution environment limitations.
