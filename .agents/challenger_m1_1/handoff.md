# Handoff Report - Milestone 1 Robustness & Correctness Challenge

## 1. Observation
We developed a test script `challenge_m1_1.py` targeting extreme edge cases, run-tested it against the `thermite` library, and analyzed the existing E2E test failures.

### Tool commands and results
1. Run the challenge script:
   ```bash
   PYTHONPATH=. .venv_homebrew/bin/python .agents/challenger_m1_1/challenge_m1_1.py
   ```
   **Output**:
   ```
   --- Running Scaler NaN/Inf Inputs ---
   StandardScaler fit NaN (expected exception): Input contains NaN or infinity values
   ...
   [PASS] Scaler NaN/Inf Inputs

   --- Running Empty Inputs ---
   StandardScaler fit empty shape (0, 2) (expected exception): Input array is empty
   ...
   [PASS] Empty Inputs

   --- Running Stratification and Imbalanced Classes ---
   Testing train_test_split stratify with a class of size 1...
   Split succeeded! Train size: 7, Test size: 3
   y_train: [0 0 0 0 0 0 1]
   y_test: [0 0 0]
   Highly imbalanced split succeeded!
   [PASS] Stratification and Imbalanced Classes

   --- Running Shuffling Randomness Reproducibility ---
   Same seed produces identical splits: Verified.
   ...
   [PASS] Shuffling Randomness Reproducibility

   --- Running Unknown Categories in Encoders ---
   LabelEncoder transform unknown category (expected exception): y contains previously unseen labels: 4
   ...
   [PASS] Unknown Categories in Encoders

   All challenge tests executed successfully!
   ```

2. Run the existing E2E preprocessing tests:
   ```bash
   PYTHONPATH=. .venv_homebrew/bin/python -m pytest tests/test_tier1_preprocessing.py tests/test_tier2_preprocessing.py
   ```
   **Output**:
   ```
   FAILED tests/test_tier1_preprocessing.py::test_standard_scaler_no_mean - AssertionError
   FAILED tests/test_tier2_preprocessing.py::test_min_max_scaler_single_row - AssertionError
   FAILED tests/test_tier2_preprocessing.py::test_label_encoder_float_values - AssertionError
   ...
   ```

### Verbatim Errors & File Paths
- **StandardScaler no_mean bug**:
  `crates/thermite-binding/src/lib.rs:130`
  ```rust
  if !self.core.with_mean {
      return Ok(None);
  }
  ```
  This causes `scaler.mean_` to return `None` in Python, whereas scikit-learn always returns the calculated mean.

- **MinMaxScaler scale on constant features / single row bug**:
  `crates/thermite-core/src/preprocessing.rs:206`
  ```rust
  let diff = d_max - d_min;
  if diff == 0.0 {
      scale_arr[i] = 0.0;
      min_arr[i] = min_val;
  }
  ```
  This causes `scaler.scale_` to return `0.0`, whereas scikit-learn sets `scale_` to `1.0` if the feature range is zero.

- **LabelEncoder float categories bug**:
  `thermite/preprocessing.py:108`
  ```python
  else:
      y_cast = list(y.astype(str))
      self._encoder.fit_str(y_cast)
      self.classes_ = np.array(self._encoder.get_classes_str())
  ```
  This casts any non-integer array (including floats) to string representation, returning `['0.5', '1.5', '2.5']` instead of maintaining float types.

---

## 2. Logic Chain
1. By examining the Rust source `crates/thermite-core/src/preprocessing.rs` (line 7), we observe the helper `check_finite` which rejects `NaN` and `Inf` inputs during `fit`, `transform`, and `inverse_transform` for `StandardScaler` and `MinMaxScaler`.
2. By executing `challenge_m1_1.py`, we verified that these non-finite inputs correctly propagate back to Python as `ValueError: Input contains NaN or infinity values`, satisfying the requirement to reject NaN/Inf inputs in scalers.
3. By executing the empty array test suite, we verified that empty arrays correctly raise `ValueError` on scaler fits. `train_test_split` on empty inputs also properly fails with `ValueError: Number of samples must be greater than 0`.
4. In `model_selection.rs` line 40, we see that stratification handles label splitting using Largest Remainder Method without verifying that class counts are at least 2. When run with a class containing 1 member, Thermite successfully performs the split by putting the 1 sample in the training set. This differs from scikit-learn, which raises a `ValueError` for classes of size 1 under stratification.
5. In `model_selection.rs` line 35, `SmallRng::seed_from_u64(seed)` is used if a seed is provided. We empirically verified that the same seed reproduces the exact same split, different seeds produce different splits, and `shuffle=False` is fully deterministic.
6. In `preprocessing.rs` line 437 and 481, `OneHotEncoderCore::transform_int/transform_str` correctly raises an error if `handle_unknown == "error"` and ignores the unknown categories if it is `"ignore"`. This matches scikit-learn.
7. During existing test runs, we noticed three specific bugs (StandardScaler `mean_` returning `None`, MinMaxScaler `scale_` returning `0.0` for constant features, and LabelEncoder converting float labels to strings).

---

## 3. Caveats
- Since we are operating under review-only constraints, we did not modify any source code to fix the three uncovered bugs.
- We did not verify other algorithms (e.g. Trees, Linear Models, Clustering) as they are planned for future Milestones and not yet implemented in Python wrappers.

---

## 4. Conclusion
The implementation of Milestone 1 is mostly robust under extreme inputs (NaN/Inf, empty arrays, random state reproducibility, and unknown categories in encoders).
However, three compatibility/correctness bugs and one behavior divergence were identified:
1. **Bug**: `StandardScaler.mean_` is `None` when `with_mean=False` (violates scikit-learn contract).
2. **Bug**: `MinMaxScaler.scale_` is `0.0` instead of `1.0` when feature range is 0 (violates scikit-learn contract).
3. **Bug**: `LabelEncoder` converts float targets to strings (violates type preservation contract).
4. **Divergence**: `train_test_split` with `stratify` allows splitting classes of size 1 (scikit-learn raises `ValueError`).

---

## 5. Verification Method
To independently verify:
1. **Run the challenge suite**:
   ```bash
   PYTHONPATH=. .venv_homebrew/bin/python .agents/challenger_m1_1/challenge_m1_1.py
   ```
2. **Run E2E preprocessing tests to observe the bug failures**:
   ```bash
   PYTHONPATH=. .venv_homebrew/bin/python -m pytest tests/test_tier1_preprocessing.py tests/test_tier2_preprocessing.py -k "test_standard_scaler_no_mean or test_min_max_scaler_single_row or test_label_encoder_float_values"
   ```

---

# Adversarial Review Challenge Report

## Challenge Summary
**Overall risk assessment**: MEDIUM

Thermite's Milestone 1 matches the core math and structures of scikit-learn, but fails on specific API contract details (properties and type preservation) which cause test failures and downstream model failures in pipelines.

## Challenges

### [Medium] Challenge 1: LabelEncoder Dtype Casting
- **Assumption challenged**: Non-integer labels can be converted to strings for encoding.
- **Attack scenario**: Passing an array of float labels `[0.5, 1.5, 2.5]`.
- **Blast radius**: Classes become strings, breaking downstream components that expect numeric classes or type equality.
- **Mitigation**: Implement `fit_float` / `transform_float` or dynamically manage python-native types instead of forcing all non-integers to strings.

### [Low] Challenge 2: MinMaxScaler Constant Feature Scale
- **Assumption challenged**: If range diff is 0, the scale should be 0.0.
- **Attack scenario**: Normalizing features where one column is constant.
- **Blast radius**: `scale_` is returned as 0.0, which differs from scikit-learn's `1.0`.
- **Mitigation**: In `preprocessing.rs` line 206, set `scale_arr[i] = 1.0` instead of `0.0` when `diff == 0.0`.

### [Low] Challenge 3: train_test_split Stratify Class of Size 1
- **Assumption challenged**: Largest Remainder Method works fine for all class sizes under stratification.
- **Attack scenario**: Stratifying on class labels where one class has only 1 sample.
- **Blast radius**: Split succeeds but test set has 0 samples of that class. This diverges from scikit-learn which blocks this with a `ValueError`.
- **Mitigation**: Add a validation check in `compute_split_indices` or the python wrapper to ensure all classes have at least 2 members when `stratify` is not None.

## Stress Test Results

| Scenario | Expected Behavior | Actual Behavior | Pass/Fail |
|---|---|---|---|
| NaN in StandardScaler input | Raise ValueError | Raised ValueError | PASS |
| Inf in MinMaxScaler input | Raise ValueError | Raised ValueError | PASS |
| Empty arrays in fit | Raise ValueError | Raised ValueError | PASS |
| `train_test_split` Same Seed | Identical splits | Identical splits | PASS |
| `train_test_split` Different Seed | Different splits | Different splits | PASS |
| `OneHotEncoder` handle_unknown='ignore' | Return all 0s for unknown categories | Returned all 0s | PASS |
