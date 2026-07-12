# Project: Thermite

## Architecture
Thermite is a Rust-accelerated machine learning library for Python, designed to be a drop-in replacement for scikit-learn.
It is organized as follows:
- **Rust Core Workspace**:
  - `crates/thermite-core`: Implements the ML algorithms (regression, classification, clustering, decomposition, preprocessing, etc.) using `ndarray` and parallelized via `Rayon`.
  - `crates/thermite-linalg` (optional/integrated): Shared linear algebra primitives if needed.
- **Python Bindings & Package**:
  - `thermite/`: The Python package directory containing wrappers that mimic scikit-learn module structure.
  - PyO3 bindings compiled into a private binary module (e.g., `thermite._core` or `thermite.binding`).
  - Python wrapper classes (e.g. `thermite.preprocessing.StandardScaler`) wrap the PyO3 classes, handling parameters, validating inputs, and delegating to the Rust backend.
- **E2E Testing Track**:
  - A separate, requirement-driven test suite at `tests/` or `e2e/` verifying all API surfaces and compatibility.
- **Benchmarks**:
  - Running scripts comparing Thermite vs scikit-learn.

## Code Layout
```
/Users/kartavyadikshit/Projects/Thermite/
 Cargo.toml                # Rust workspace configuration
 pyproject.toml            # Maturin and python packaging
 Cargo.lock
 crates/
    thermite-core/        # Rust ML algorithms core
       Cargo.toml
       src/
           lib.rs
           linear_model.rs
           tree.rs
           cluster.rs
           preprocessing.rs
           neighbors.rs
           metrics.rs
    thermite-binding/     # PyO3 bindings crate
        Cargo.toml
        src/
            lib.rs
 thermite/                 # Python API package
    __init__.py
    linear_model.py
    tree.py
    ensemble.py
    cluster.py
    preprocessing.py
    model_selection.py
    decomposition.py
    metrics.py
    pipeline.py
 tests/                    # E2E and unit test suites
 benchmarks/               # Performance benchmark suite
 docs/                     # Documentation files
```

## Milestones
| # | Name | Scope | Dependencies | Status | Conv ID |
|---|------|-------|--------------|--------|---------|
| 1 | Foundation & Preprocessing | Set up Cargo/Maturin workspace, implement `StandardScaler`, `MinMaxScaler`, `LabelEncoder`, `OneHotEncoder`, `train_test_split`. | None | DONE | 4f539ea2-b299-4cac-afb7-27d4a5777e72 |
| 2 | Linear Models, Neighbors & Metrics | Implement `LinearRegression`, `Ridge`, `Lasso`, `LogisticRegression`, `KNeighborsClassifier`, and all metrics. | M1 | DONE | TBD |
| 3 | Trees, Clustering & PCA | Implement `DecisionTree*`, `RandomForest*`, `GradientBoosting*`, `KMeans`, `DBSCAN`, `PCA` (with Rayon). | M1 | DONE | TBD |
| 4 | Pipeline & Model Selection | Implement `Pipeline`, `cross_val_score`, `GridSearchCV`. | M1, M2, M3 | DONE | TBD |
| 5 | E2E Integration & Hardening | Final verification of 100% E2E test pass + adversarial hardening (Tier 5). | M1, M2, M3, M4, E2E | DONE | TBD |
| E2E | E2E Testing Track | Requirement-driven opaque-box test suite (Tiers 1-4) & runner. Publishes `TEST_READY.md`. | None | DONE | 2be0998b-3422-4735-8651-607c24e87f4a |
| F0 | Exploration & Architecture | Research code architecture, NaN points, SVM FFI, and BLAS config. | None | DONE | fc97c6b3-f083-4036-98f1-c39e0a8cfa0b |
| F1 | NaN Support | Implement learned NaN routing in trees and mean-imputation in linear models. | F0 | DONE | a824cedd-5307-4b7b-aa98-d18e132fd0c3 |
| F2 | Kernel SVM | Wrap/compile LIBSVM via C-bindings and implement SVC. | F0 | DONE | 2ab12a29-fb0d-4135-aa07-513a62a4157c |
| F3 | BLAS/MKL Linkage | Configure dynamic BLAS/MKL feature flags in Cargo.toml. | F0 | DONE | 2da9c2e2-0084-4534-86a5-21433344c97c |
| F4 | Final Integration & Audit | E2E tests, Challenger verification, and Forensic Audit. | F1, F2, F3 | PLANNED | TBD |

## Interface Contracts
### Python Wrappers  PyO3 Rust Bindings (`thermite._core`)
- Inputs are validated NumPy arrays. PyO3 bindings accept NumPy arrays (using `numpy-rust` or similar, e.g., `PyReadonlyArray2` for features $X$ and `PyReadonlyArray1` for targets $y$).
- Outputs are returned as NumPy arrays (e.g., `PyArray2`, `PyArray1`).
- All class parameters (e.g. `n_estimators`, `max_depth`, `n_jobs`) are passed from Python wrappers to PyO3 classes upon initialization or `fit`.
- If an operation fails, Rust code propagates errors back to Python as appropriate Python exceptions (e.g., `ValueError`, `TypeError`).
