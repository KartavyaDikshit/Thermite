# Thermite Follow-up Implementation Plan

This document outlines the milestones and tracking for implementing the new requirements in the follow-up request (2026-07-12T17:45:09Z).

## Milestones

### Milestone F0: Exploration & Architecture
- **Goal**: Research current linear model and tree implementations, identify best integration points for NaN handling, SVM bindings (using LIBSVM/LIBLINEAR or wrapping a C library), and BLAS/MKL linkage options.
- **Verification**: Exploration reports with specific code recommendations.
- **Dependencies**: None.

### Milestone F1: NaN / Missing Data Support
- **Goal**: Implement dynamic NaN handling.
  - **Trees**: Branch-splitting routing for NaNs (DecisionTreeClassifier/Regressor, and ensure ensemble models wrap them correctly).
  - **Linear Models**: Simple imputation (e.g. mean/median imputation during fit/predict/transform, in StandardScaler/MinMaxScaler or inside LogisticRegression and LinearRegression models).
- **Verification**: Complete implementation + `test_nan_support.py` passes with >90% accuracy on a dummy dataset containing NaNs.
- **Dependencies**: Milestone F0.

### Milestone F2: Kernel SVM via C-bindings
- **Goal**: Wrap/integrate optimized SVM solvers (like LIBSVM/LIBLINEAR) via C-bindings. Implement `SVC` with `rbf` and `poly` kernels in `thermite.svm` (mimicking scikit-learn's `SVC`).
- **Verification**: `test_svm.py` successfully trains and predicts using `SVC(kernel='rbf')` and `SVC(kernel='poly')` on a dummy dataset, matching the scikit-learn API.
- **Dependencies**: Milestone F0.

### Milestone F3: Dynamic BLAS/MKL Linkage
- **Goal**: Configure Cargo.toml and ndarray/gemm options to optionally link with optimized hardware BLAS libraries (e.g., Apple Accelerate, Intel MKL).
- **Verification**: `cargo test --features mkl` (or equivalent) compiles and links cleanly.
- **Dependencies**: Milestone F0.

### Milestone F4: Final E2E Integration and Adversarial Hardening
- **Goal**: Run comprehensive testing suite, run Forensic Auditor, run Challenger to perform adversarial stress-testing.
- **Verification**: Verification and validation checkmarks pass.
- **Dependencies**: Milestones F1, F2, F3.

