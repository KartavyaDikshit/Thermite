# Handoff Report — Milestone 1 Quality and Adversarial Review

## 1. Quality Review Report

### Review Summary
**Verdict**: REQUEST_CHANGES

The Milestone 1 implementation compiles successfully and passes pure Rust unit tests and a simple Python verification script. However, detailed inspection and E2E compatibility checks against the `pytest` test suite reveal multiple critical and major discrepancies from the `scikit-learn` interface and behavior. The current implementation is NOT a drop-in replacement.

---

### Findings

#### [Critical] Finding 1: `train_test_split` order discrepancy when `shuffle=False`
- **What**: When `shuffle=False` is passed, `train_test_split` returns test split indices first, followed by train split indices. In `scikit-learn`, the first part of the array is returned as train and the second part as test.
- **Where**: `crates/thermite-core/src/model_selection.rs` line 146:
  ```rust
  let (test_part, train_part) = indices.split_at(n_test);
  ```
- **Why**: Returning test data before train data causes mismatch and corrupts sequential split assumptions (e.g. training on future, testing on past).
- **Suggestion**: Change the split logic so that when `shuffle=False` (or generally), the indices are sliced such that `train_indices` correspond to the first `n_train` elements, and `test_indices` correspond to the subsequent `n_test` elements. E.g.:
  ```rust
  let (train_part, test_part) = indices.split_at(n_train);
  ```

#### [Major] Finding 2: `LabelEncoder` coerces float labels to string classes
- **What**: Fitting `LabelEncoder` on a 1D float array yields a `classes_` array containing strings (e.g. `['0.5', '1.5', '2.5']`) instead of floats (`[0.5, 1.5, 2.5]`).
- **Where**: `thermite/preprocessing.py` line 109:
  ```python
  y_cast = list(y.astype(str))
  self._encoder.fit_str(y_cast)
  ```
- **Why**: Scikit-learn preserves the float type of classes, and downstream operations expecting numeric classes will raise type errors.
- **Suggestion**: Add a `fit_float` and `transform_float` option/binding to the Rust core and PyO3 module, or dynamically determine float representation in Python.

#### [Major] Finding 3: `MinMaxScaler` returns incorrect scale and min values on constant/single-row inputs
- **What**: Fitting `MinMaxScaler` on a single row (where `diff == 0.0`) results in `scale_` being `[0.0, 0.0]` and `min_` being `[0.0, 0.0]`.
- **Where**: `crates/thermite-core/src/preprocessing.rs` line 206:
  ```rust
  if diff == 0.0 {
      scale_arr[i] = 0.0;
      min_arr[i] = min_val;
  }
  ```
- **Why**: Scikit-learn sets `scale_` to `1.0` and `min_` to `feature_range[0] - data_min * 1.0` for constant features.
- **Suggestion**: Align the constant feature handling in Rust to:
  ```rust
  if diff == 0.0 {
      scale_arr[i] = 1.0;
      min_arr[i] = min_val - d_min * 1.0;
  }
  ```

#### [Major] Finding 4: `StandardScaler.mean_` is `None` when `with_mean=False`
- **What**: Fitting `StandardScaler` with `with_mean=False` returns `None` for the `mean_` attribute.
- **Where**: `crates/thermite-binding/src/lib.rs` line 130:
  ```rust
  if !self.core.with_mean {
      return Ok(None);
  }
  ```
- **Why**: In scikit-learn, the mean is always computed and populated in `mean_` regardless of whether `with_mean` is True or False.
- **Suggestion**: Remove the `if !self.core.with_mean` guard in the PyO3 getter for `mean_`.

#### [Major] Finding 5: `OneHotEncoder` missing standard parameters in wrapper
- **What**: `OneHotEncoder.__init__` does not accept `sparse_output`, `drop`, or `categories` arguments, leading to `TypeError` in E2E tests.
- **Where**: `thermite/preprocessing.py` line 139:
  ```python
  class OneHotEncoder:
      def __init__(self, *, handle_unknown="error"):
  ```
- **Why**: Standard E2E tests and client applications using drop options or custom categories cannot initialize the class.
- **Suggestion**: Implement wrapper logic and matching Rust core/binding methods for `drop="first"` (or other drop arrays) and custom categories.

---

### Verified Claims
- **Claim**: Crate compiles cleanly and basic functions link -> Verified via `.venv/bin/python verify_m1.py` -> PASS.
- **Claim**: Rust core unit tests -> Verified via `cargo test` -> PASS (7 tests pass).
- **Claim**: Basic StandardScaler, MinMaxScaler, LabelEncoder, and OneHotEncoder fit/transform functionality under standard inputs -> Verified via `pytest` -> PASS.

---

### Coverage Gaps
- Float datasets in `LabelEncoder` were not verified by the implementation agent.
- Parameters `sparse_output`, `drop`, `categories` for `OneHotEncoder` were not implemented.
- Deterministic behavior of `train_test_split(shuffle=False)` was not verified against scikit-learn.

---

## 2. Adversarial Challenge Report

### Challenge Summary
**Overall risk assessment**: HIGH

The current implementation contains multiple logical and interface mismatches which will break E2E integration and real-world pipelines.

---

### Challenges

#### [High] Challenge 1: `train_test_split(shuffle=False)` Reversal
- **Assumption challenged**: That splitting order (test first vs train first) is irrelevant.
- **Attack scenario**: Split sequential logs or time-series data for train/test. Because the test slice is returned first, the train set gets the latter part of the sequence and test gets the early part. This flips temporal ordering, leading to training on future data.
- **Blast radius**: Complete logic corruption of time-series model validation.
- **Mitigation**: Correct indices partitioning order.

#### [Medium] Challenge 2: Label Encoder type coercion
- **Assumption challenged**: That float labels can be mapped to strings in `classes_`.
- **Attack scenario**: Code doing class index mapping or math checks on class labels (e.g. `if val in classes_`) fails due to type mismatch (float vs string).
- **Blast radius**: Downstream classification pipelines.
- **Mitigation**: Keep original type in `classes_`.

---

## 3. 5-Component Handoff Report

### Observation
* **Rust Unit Tests**: `cargo test` succeeded with 7 tests.
* **Verification Script**: `.venv/bin/python verify_m1.py` printed `All Python verification tests pass!`.
* **Pytest Errors**: Running `PYTHONPATH=. .venv_homebrew/bin/pytest tests/test_infra_check.py tests/test_tier1_preprocessing.py tests/test_tier2_preprocessing.py` raised multiple errors:
  - `AssertionError` in `test_min_max_scaler_single_row` (scale actual `[0.0, 0.0]`, desired `[1.0, 1.0]`).
  - `AssertionError` in `test_label_encoder_float_values` (classes actual `['0.5', '1.5', '2.5']`, desired `[0.5, 1.5, 2.5]`).
  - `TypeError` in `OneHotEncoder` due to missing `sparse_output`, `drop`, `categories` keyword arguments.
  - `AssertionError` in `test_standard_scaler_no_mean` (scaler.mean_ is None).

### Logic Chain
1. A drop-in replacement must match the scikit-learn public API signature and attributes.
2. The current python wrappers and Rust code fail to handle constant inputs (`MinMaxScaler`), float categories (`LabelEncoder`), disabled mean tracking (`StandardScaler`), and standard options (`OneHotEncoder`).
3. As a result, 14 E2E tests fail under the thermite backend.

### Caveats
- Benchmarks and performance tests under high Rayon concurrency have not been stress-tested.
- Platform is limited to macOS architecture.

### Conclusion
The verdict is `REQUEST_CHANGES`. The preprocessing and model selection components have significant functional gaps and behavior mismatches.

### Verification Method
Run the following verification command in the project root:
```bash
PYTHONPATH=. .venv_homebrew/bin/pytest tests/test_infra_check.py tests/test_tier1_preprocessing.py tests/test_tier2_preprocessing.py
```
Expected: All tests pass. Current: 14 failed, 27 passed.
