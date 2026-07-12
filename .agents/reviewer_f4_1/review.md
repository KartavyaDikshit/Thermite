# Quality & Adversarial Review Report

## Review Summary

**Verdict**: REQUEST_CHANGES

This verdict is due to robustness risks (potential panics on non-contiguous array inputs in the PyO3 wrappers) and cargo feature compilation issues for `openblas`/`intel-mkl` BLAS linkages. The core algorithms for learned NaN routing, mean imputation, and SVM SMO solver are logically sound and correct.

---

## Findings

### Major Finding 1: Panic on Non-Contiguous Target Vectors in Tree Bindings

- **What**: Potential panic due to `.unwrap()` on `as_slice()` for target array `y`.
- **Where**: `crates/thermite-binding/src/tree_bind.rs` (lines 43, 102):
  ```rust
  self.core.fit(&x_view, y_view.as_slice().unwrap());
  ```
- **Why**: `y_view.as_slice()` returns `Some(&[f64])` ONLY if the array is contiguous. If a non-contiguous array (e.g., a slice `y[::2]`) is passed, `as_slice()` returns `None`, and `.unwrap()` causes a panic that crashes the Python process.
- **Suggestion**: Accept the array view directly in `CoreDecisionTree*::fit` or use `y_view.to_slice()` to copy it if not contiguous.

### Major Finding 2: Incompatible build-dependency in `openblas-src` Crate

- **What**: Compilation failure when building with `--features openblas`.
- **Where**: `crates/thermite-core/Cargo.toml` (lines 14, 19)
- **Why**: Enabling the `openblas` feature pulls in `openblas-src` version 0.10, which pulls in `openblas-build` 0.10.16. This build dependency fails to compile due to TLS feature gates in the `ureq` crate on the system.
- **Suggestion**: Upgrade `openblas-src` to a newer version that resolves the `ureq` TLS feature conflict, or document system prerequisites.

### Major Finding 3: Missing Feature Forwarding in Maturin Crate

- **What**: Features like `intel-mkl`, `openblas`, and `accelerate` are not exposed at the binding level.
- **Where**: `crates/thermite-binding/Cargo.toml`
- **Why**: Since `thermite-binding` is the workspace target for Maturin, users cannot pass `--features accelerate` to Maturin directly unless forwarding features are defined in `crates/thermite-binding/Cargo.toml` to delegate to `thermite-core`.
- **Suggestion**: Add the following feature section to `crates/thermite-binding/Cargo.toml`:
  ```toml
  [features]
  intel-mkl = ["thermite-core/intel-mkl"]
  openblas = ["thermite-core/openblas"]
  accelerate = ["thermite-core/accelerate"]
  ```

### Minor Finding 4: Simplified OVO Probability Coupling in SVC

- **What**: Non-standard probability coupling for multiclass SVC.
- **Where**: `crates/thermite-core/src/svm.rs` (lines 292-310)
- **Why**: Scikit-learn's `SVC.predict_proba` solves a quadratic optimization problem to couple pairwise probabilities (Wu, Lin, Weng 2004). The implementation in `svm.rs` uses a simplified summing heuristic (`s[idx_i] / sum_s`), which is not fully compatible/faithful to scikit-learn.
- **Suggestion**: Document this difference or implement the iterative Wu, Lin, Weng coupling.

---

## Verified Claims

- **Learned NaN routing in Decision Trees** → verified via inspecting `tree.rs` and running `test_nan_support.py` → **PASS**
- **Mean Imputation in Linear Models** → verified via inspecting `linear_model.rs` and running `test_nan_support.py` → **PASS**
- **C++ SMO SVM Solver and Platt Scaling** → verified via inspecting `svm.cpp`/`svm.rs` and running `test_svm.py` → **PASS**
- **Accelerate BLAS Linkage on macOS** → verified via running `cargo test --features accelerate` → **PASS**

---

## Coverage Gaps

- **Non-contiguous inputs in other estimators** — risk level: **Medium** — recommendation: Auditing other bindings (e.g. `cluster_bind.rs`, `neighbors_bind.rs`) for potential `.unwrap()` on `.as_slice()` or other unsafe array conversions.

---

## Challenge Report (Adversarial Critique)

**Overall risk assessment**: MEDIUM

### High Challenge 1: Out-of-bounds / Panics via Sliced Python Inputs

- **Assumption challenged**: The target vector `y` passed from Python is contiguous.
- **Attack scenario**: Python caller runs `clf.fit(X, y[::2])` where `y` is sliced.
- **Blast radius**: `y_view.as_slice()` returns `None`, leading to a Rust panic and immediate Python process crash.
- **Mitigation**: Replace `.as_slice().unwrap()` with a fallback that copies non-contiguous views to a contiguous vec, or modify `CoreDecisionTree` to accept array views instead of slices.

### Medium Challenge 2: Training Bias / Overfitting in Platt Scaling Calibration

- **Assumption challenged**: Calibrating Platt scaling parameters on training decision values yields calibrated test-time probabilities.
- **Attack scenario**: The SVM SMO optimizer achieves high accuracy on training data, causing training decision values to be large. The fitted sigmoid parameters $A$ and $B$ will be overfitted, yielding overconfident probabilities on test data.
- **Blast radius**: Poorly calibrated probability estimates (`predict_proba`) in production.
- **Mitigation**: Use internal cross-validation (like scikit-learn's 5-fold CV) to generate out-of-fold decision values for Platt scaling.
