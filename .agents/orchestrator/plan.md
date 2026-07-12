# Thermite Implementation Plan

This document outlines the milestones and tracking for the implementation of Thermite.

## Execution Tracks

We run two parallel execution tracks:
1. **E2E Testing Track**: Build a robust, scikit-learn-compatible E2E test suite covering Tiers 1-4.
2. **Implementation Track**: Progressively build the Rust core and Python bindings through 4 implementation milestones, ending with a final E2E verification and adversarial hardening milestone.

## Milestones

### Milestone E2E: E2E Testing Track
- **Goal**: Create comprehensive requirement-driven test suite (Tiers 1-4) using pytest.
- **Verification**: Outputs `TEST_READY.md` containing runner command and tier counts.
- **Dependencies**: None.

### Milestone 1: Foundation & Preprocessing
- **Goal**: Set up Cargo workspace, PyO3 bindings, and python package scaffolding. Implement preprocessing (`StandardScaler`, `MinMaxScaler`, `LabelEncoder`, `OneHotEncoder`) and utility `train_test_split`.
- **Verification**: Successful build, `pip install -e .` installation, and execution of basic preprocessing/split tests.
- **Dependencies**: None.

### Milestone 2: Linear Models, Neighbors & Metrics
- **Goal**: Implement `LinearRegression`, `Ridge`, `Lasso`, `LogisticRegression`, `KNeighborsClassifier`, and performance metrics (`accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `roc_auc_score`, `mean_squared_error`, `r2_score`).
- **Verification**: Algorithm correctness tests.
- **Dependencies**: Milestone 1.

### Milestone 3: Trees, Clustering & PCA
- **Goal**: Implement decision trees, random forests, gradient boosting, clustering (`KMeans`, `DBSCAN`), and dimensionality reduction (`PCA`). Use Rayon for parallelism where performance wins are needed.
- **Verification**: Algorithm correctness and performance tests.
- **Dependencies**: Milestone 1.

### Milestone 4: Pipeline & Model Selection
- **Goal**: Implement `Pipeline` chaining, `cross_val_score`, and `GridSearchCV`.
- **Verification**: End-to-end grid search and pipeline execution tests.
- **Dependencies**: Milestones 1, 2, 3.

### Milestone 5: E2E Integration and Adversarial Hardening (Final Milestone)
- **Goal**: Pass 100% of the E2E test suite (Tiers 1-4) and run Phase 2 (Tier 5 adversarial hardening and coverage checks).
- **Verification**: Complete test pass, no gaps, clippy/ruff lints pass, doc generation, update `DEVLOG.md` and `README.md`.
- **Dependencies**: Milestones 1, 2, 3, 4, and Milestone E2E.
