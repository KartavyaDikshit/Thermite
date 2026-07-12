## 2026-07-12T19:54:14+02:00

Please implement Kernel SVM Support via C-bindings in Thermite.
Your working directory is /Users/kartavyadikshit/Projects/Thermite/.agents/worker_f2_svm.

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A Forensic Auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Tasks:
1. **C++ SVM Solver** (`crates/thermite-core/src/libsvm/svm.h` and `crates/thermite-core/src/libsvm/svm.cpp`):
   - Implement a self-contained Sequential Minimal Optimization (SMO) SVM solver in C++.
   - Support RBF kernel: exp(-gamma * ||x_i - x_j||^2) and Polynomial kernel: (gamma * <x_i, x_j> + coef0)^degree.
   - Expose C-compatible training and prediction interfaces (e.g. `svm_train` and `svm_predict` or custom equivalents matching libsvm structs).
2. **Build Configuration** (`crates/thermite-core/Cargo.toml` and `crates/thermite-core/build.rs`):
   - Add `cc = "1.2.66"` to `[build-dependencies]` in `crates/thermite-core/Cargo.toml`.
   - Create `crates/thermite-core/build.rs` compiling `src/libsvm/svm.cpp` using `cc::Build`.
3. **Rust SVM Module** (`crates/thermite-core/src/svm.rs` and `crates/thermite-core/src/lib.rs`):
   - Declare the FFI signatures for the C++ functions and structs.
   - Implement a safe Rust wrapper/struct `SVC` with `fit`, `predict`, and `predict_proba`.
   - Manage memory carefully using standard FFI practices (like `Drop` to release pointers).
4. **PyO3 Bindings & Python Package** (`crates/thermite-binding/src/svm_bind.rs`, `crates/thermite-binding/src/lib.rs`, `thermite/svm.py`, and `thermite/__init__.py`):
   - Create PyO3 bindings for `SVC` in `svm_bind.rs` and register the class in `lib.rs`.
   - Write a scikit-learn-compatible Python wrapper `SVC` in `thermite/svm.py` (inheriting from sklearn base class if applicable, or mimicking it completely with `fit`, `predict`, `predict_proba` methods and parameters: `C`, `kernel`, `degree`, `gamma`, `coef0`, `probability`).
5. **Verification**:
   - Create `tests/test_svm.py` to train `SVC(kernel='rbf')` and `SVC(kernel='poly')` on a dummy classification dataset (like blobs or digits), and verify that fit/predict/predict_proba run successfully without exceptions and are correct.
   - Verify that `cargo test` and `pytest tests/test_svm.py` pass.
   - Write a detailed `handoff.md` with your changes and test outputs in your directory.
