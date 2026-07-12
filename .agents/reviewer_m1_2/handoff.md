# Handoff Report  Milestone 1 Review (Reviewer 2)

## 1. Observation

We directly observed and verified the following:
* **Happy Path Verification**: 
  * Running `cargo test` succeeds with all 7 tests passing.
  * Running `.venv/bin/python verify_m1.py` succeeds and prints `"All Python verification tests pass!"`.
* **E2E Test Suite Run**: 
  * Running the preprocessing E2E test suite via:
    `PYTHONPATH=.venv/lib/python3.14/site-packages:. pytest tests/test_infra_check.py tests/test_tier1_preprocessing.py tests/test_tier2_preprocessing.py`
    results in **14 failures** out of 41 test cases (27 passed, 14 failed).
  * Running the same tests against scikit-learn via `USE_SKLEARN=1` results in **41 passed** test cases.
* **Code Audits**:
  * **StandardScaler**: Exposes `mean_` as `None` when `with_mean=False` even if `with_std=True` (in `crates/thermite-binding/src/lib.rs` line 130). Scikit-learn exposes `mean_` in this case because it is calculated during variance computation.
  * **MinMaxScaler**: In `crates/thermite-core/src/preprocessing.rs` line 206, when a feature has zero variance (constant column), `scale_` is set to `0.0` and `min_` is set to `feature_range.0`. Scikit-learn sets `scale_` to `range_diff` (e.g. `1.0`) and `min_` to `feature_range.0 - data_min * scale_`.
  * **LabelEncoder**: In `thermite/preprocessing.py` line 104, float inputs are cast to string representation, resulting in a string array for `classes_` instead of keeping float types.
  * **OneHotEncoder**: The constructor in `thermite/preprocessing.py` line 140 only accepts `handle_unknown` and does not accept `sparse_output`, `drop`, or `categories` arguments, leading to `TypeError` exceptions during E2E test suite calls. Also, empty arrays do not raise `ValueError` on `fit()`, and calling `inverse_transform` on columns with zero categories can cause an index out-of-bounds panic (`sub_slice[0]`) in `crates/thermite-core/src/preprocessing.rs` lines 524 and 564.
  * **train_test_split**: In `crates/thermite-core/src/model_selection.rs` line 65, the stratified splitting logic iterates over a standard `HashMap` (`label_to_indices.iter()`) without sorting the keys. Because Rust's `HashMap` has randomized iteration order, this introduces non-determinism in the split indices when `shuffle=False`.

---

## 2. Logic Chain

1. The project's E2E test suite (`tests/`) defines the scikit-learn API compatibility contracts for Thermite.
2. The E2E tests pass 100% when run against the `scikit-learn` backend (`USE_SKLEARN=1`), confirming the validity of the tests.
3. Running the same tests on the `thermite` backend results in 14 failures in the preprocessing modules.
4. Therefore, the current Milestone 1 implementation deviates from the required scikit-learn API contract in several correctness and completeness edge cases.
5. In addition, the Rust core contains potential panics and non-deterministic behavior that must be resolved.

---

## 3. Caveats

* Future modules (e.g. `linear_model`, `metrics`, `tree`, `cluster`, `pipeline`) are not yet implemented as they are scheduled for later milestones. Collection failures for files referencing these modules are expected and ignored for this review.
* Gaps in `OneHotEncoder` parameters (`drop`, `categories`, `sparse_output`) were explicitly omitted from the worker's M1 scope but are required to pass the E2E test suite.

---

## 4. Conclusion & Verdict

**Verdict**: **REQUEST_CHANGES**

### Quality Review Report

#### Finding 1 (Critical) - OneHotEncoder Interface Incompleteness
* **What**: `OneHotEncoder` constructor is missing `sparse_output`, `drop`, and `categories` parameters.
* **Where**: `thermite/preprocessing.py:140`, `crates/thermite-binding/src/lib.rs:320`, `crates/thermite-core/src/preprocessing.rs:387`.
* **Why**: E2E tests call these parameters; their absence causes `TypeError` crashes.
* **Suggestion**: Add the parameters to the Python wrapper, PyO3 bindings, and Rust core logic.

#### Finding 2 (Major) - MinMaxScaler Constant Columns Mismatch
* **What**: `MinMaxScaler` sets `scale_ = 0.0` for constant features.
* **Where**: `crates/thermite-core/src/preprocessing.rs:206`.
* **Why**: Scikit-learn sets `scale_ = range_diff` (e.g., 1.0) and `min_ = feature_range[0] - data_min * scale_`.
* **Suggestion**: Align the constant feature scaling logic with scikit-learn's behavior.

#### Finding 3 (Major) - StandardScaler Hidden Mean Attribute
* **What**: `StandardScaler.mean_` returns `None` if `with_mean=False` even if `with_std=True`.
* **Where**: `crates/thermite-binding/src/lib.rs:130`.
* **Why**: The mean is computed during variance calculation and must be exposed.
* **Suggestion**: Return the computed mean if `with_mean` or `with_std` is true.

#### Finding 4 (Minor) - LabelEncoder Dtype Coercion
* **What**: Float arrays are converted to string arrays.
* **Where**: `thermite/preprocessing.py:104`.
* **Why**: `classes_` contains strings instead of floats.
* **Suggestion**: Keep float array categories as floats or support float/double arrays in the Rust core encoder.

---

### Adversarial Challenge Report

**Overall risk assessment**: **HIGH**

#### Challenge 1 (High) - Rust Panic on Zero Category Slicing
* **Assumption challenged**: Columns always have at least one category during inverse transform.
* **Attack scenario**: Call `inverse_transform` on data where `fit` encountered an empty column (or empty categories list).
* **Blast radius**: Panic at `sub_slice[0]` causing the Python interpreter to crash.
* **Mitigation**: Add a length check on `cats` before indexing `sub_slice[0]`.

#### Challenge 2 (Medium) - Non-deterministic Splits with shuffle=False
* **Assumption challenged**: Splits are reproducible and deterministic for a given input if `shuffle=False`.
* **Attack scenario**: Run stratified `train_test_split` with `shuffle=False` on the same input multiple times.
* **Blast radius**: The returned training and test indices will be in a non-deterministic order due to standard `HashMap` iteration.
* **Mitigation**: Sort the keys of `label_to_indices` (or use a `BTreeMap` / collect and sort keys) before computing class splits.

---

## 5. Verification Method

To verify the E2E preprocessing failures:
1. Run cargo tests:
   ```bash
   cargo test
   ```
2. Run Python verification:
   ```bash
   .venv/bin/python verify_m1.py
   ```
3. Run Python E2E preprocessing tests (expect 14 failures on the thermite backend):
   ```bash
   PYTHONPATH=.venv/lib/python3.14/site-packages:. pytest tests/test_infra_check.py tests/test_tier1_preprocessing.py tests/test_tier2_preprocessing.py
   ```
4. Run Python E2E preprocessing tests against scikit-learn to confirm E2E test correctness:
   ```bash
   PYTHONPATH=.venv/lib/python3.14/site-packages:. USE_SKLEARN=1 pytest tests/test_infra_check.py tests/test_tier1_preprocessing.py tests/test_tier2_preprocessing.py
   ```
