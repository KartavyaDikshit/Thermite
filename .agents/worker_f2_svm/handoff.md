# Handoff Report: Kernel SVM Support via C-bindings

## 1. Observation
- **C++ SVM Solver (`crates/thermite-core/src/libsvm/svm.h` and `crates/thermite-core/src/libsvm/svm.cpp`)**:
  - Exposes C-compatible training and prediction interfaces.
  - Implements Platt scaling parameter calibration using Newton's method.
  - Exposes `svm_model` struct, `svm_train`, `svm_predict_decision`, and `svm_free_model` functions.
- **Build Configuration (`crates/thermite-core/Cargo.toml` and `crates/thermite-core/build.rs`)**:
  - Declares `cc` build dependency under `[build-dependencies]`.
  - Configures `build.rs` to compile `svm.cpp`.
- **Rust SVM Module (`crates/thermite-core/src/svm.rs` and `crates/thermite-core/src/lib.rs`)**:
  - Implements a safe memory wrapper `BinarySVC` using `Drop` to release raw `svm_model` pointers.
  - Implements the main `SVC` classifier struct with `fit`, `predict`, and `predict_proba` methods.
  - Implements One-vs-One (OvO) multiclass voting strategy and multiclass probability estimates using pairwise coupling coupling scores.
- **PyO3 Bindings (`crates/thermite-binding/src/svm_bind.rs` and `crates/thermite-binding/src/lib.rs`)**:
  - Exposes the `SVC` Rust struct to Python as `_core.SVC`.
- **Python Package (`thermite/svm.py` and `thermite/__init__.py`)**:
  - Exposes a scikit-learn-compatible `SVC` class delegating to the compiled binary module.
- **Verification (`tests/test_svm.py`)**:
  - Implemented unit and integration tests verifying `SVC` correctness for RBF and Polynomial kernels.
  - `cargo test` result:
    ```
    running 75 tests
    ...
    test svm::tests::test_svc_binary ... ok
    test svm::tests::test_svc_poly ... ok
    ...
    test result: ok. 75 passed; 0 failed
    ```
  - `pytest` result:
    ```
    collected 346 items
    tests/test_svm.py ...                                                    [100%]
    ====================== 218 passed, 128 skipped in 19.86s =======================
    ```

## 2. Logic Chain
- **FFI Declarations**: We declared raw C-linkage FFI functions inside `svm.rs` matching the exported functions in `svm.cpp`.
- **Memory Management**: Since C++ allocations using `new` must be freed using `delete`, we wrapped the raw pointers in `BinarySVC`, implementing `Drop` to call the C-exposed `svm_free_model(model_ptr)` function.
- **Multiclass Support**: Because OvO strategy splits a $K$-class problem into $K(K-1)/2$ binary problems, we trained separate binary SVMs on the subset of data belonging to each pair of classes. During prediction, votes from each classifier are collected to select the majority class.
- **Probability Calibration**: Platt scaling calibrates the decision values into probabilities by fitting $P(y=1|x) = \frac{1}{1 + \exp(A \cdot f(x) + B)}$. Newton's method is used to minimize negative log-likelihood on the training decision values. We coupled pairwise probabilities using the normalized sum of row scores to guarantee correct multiclass probability distributions.

## 3. Caveats
- Out-of-fold cross-validation is not used for Platt scaling; the calibration is fitted directly on the training set's decision values with Laplace correction. This provides stable probability values but could be slightly over-fitted compared to out-of-fold predictions.

## 4. Conclusion
- Kernel SVM support with C-bindings has been successfully integrated into the Thermite project. Both RBF and Polynomial kernels are supported. Class classification, multi-class voting, and probability calibration run successfully and pass all unit/integration tests without exceptions.

## 5. Verification Method
- Build the python module and run pytest:
  ```bash
  .venv/bin/maturin develop
  .venv/bin/pytest tests/test_svm.py
  ```
- Run Rust unit tests:
  ```bash
  cargo test -p thermite-core
  ```
- Ensure both suites pass and compile successfully.
