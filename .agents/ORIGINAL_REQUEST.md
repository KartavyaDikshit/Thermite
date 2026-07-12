# Original User Request

## Initial Request  2026-07-12T11:20:59Z

Build **Thermite**  a production-ready, Rust-accelerated machine learning library for Python that is API-compatible with scikit-learn. The Rust core implements ML algorithms with automatic parallelism via Rayon, exposed to Python via PyO3 bindings. Users should be able to `pip install thermite-ml` and swap `from sklearn`  `from thermite` with zero code changes. This is a serious open-source project with startup/commercial ambitions.

Working directory: /Users/kartavyadikshit/Projects/Thermite
Integrity mode: development

**Important**: The repository is already initialized with a README.md, DEVLOG.md, and .gitignore on the `main` branch. Build on top of the existing files  do not overwrite or delete them. Update DEVLOG.md with a work log entry at the end of the build documenting what was done, decisions made, and any issues encountered.

## Requirements

### R1. Rust ML Core
Build a Rust workspace with ML algorithm implementations that are correct and significantly faster than scikit-learn's equivalents. Use the ndarray/nalgebra ecosystem for linear algebra. Use Rayon for data parallelism. The following algorithms must be implemented with full `fit`/`predict`/`transform` semantics:

**Linear Models**: LinearRegression, Ridge, Lasso, LogisticRegression
**Tree-Based**: DecisionTreeClassifier, DecisionTreeRegressor, RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
**Clustering**: KMeans, DBSCAN
**Decomposition**: PCA
**Preprocessing**: StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
**Neighbors**: KNeighborsClassifier
**Model Selection**: train_test_split, cross_val_score, GridSearchCV
**Metrics**: accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error, r2_score
**Pipeline**: Pipeline (sklearn-compatible chaining of transformers and estimators)

### R2. Python API via PyO3
Expose all Rust algorithms as a Python package (`thermite`) via PyO3 and maturin. The Python API must:
- Accept numpy arrays as input (X, y) and return numpy arrays
- Mirror scikit-learn's class names, method signatures, and parameter names
- Support `n_jobs` parameter for parallelism control where sklearn does
- Include proper Python type hints and docstrings
- Be installable via `pip install -e .` using maturin develop

### R3. Benchmarks Suite
Build an automated benchmark suite that compares Thermite vs scikit-learn head-to-head on:
- All implemented algorithms
- Multiple dataset sizes (1K, 10K, 100K, 1M rows where feasible)
- Both `fit()` and `predict()` operations
- Results output as a markdown table and JSON

The benchmark script must be runnable via a single command (e.g., `python benchmarks/run_all.py`).

### R4. Documentation
- Update the existing README.md with accurate installation instructions, usage examples, and benchmark results
- Create API reference documentation (can use pdoc, mkdocs, or similar)
- Include a "Migration from scikit-learn" guide showing side-by-side code comparisons

### R5. CI/CD and Quality
- GitHub Actions workflow for: Rust tests, Python tests, benchmarks, linting (clippy + ruff)
- pyproject.toml configured for PyPI publishing via maturin
- At least 80% test coverage on Rust core algorithms (unit tests with known expected outputs)
- Python integration tests that verify sklearn API compatibility (same inputs  equivalent outputs within floating-point tolerance)

## Acceptance Criteria

### Functional Correctness
- [ ] All algorithms listed in R1 are implemented and pass unit tests
- [ ] Python integration tests verify that for each algorithm, given the same input data, Thermite produces outputs equivalent to scikit-learn (within 1e-6 tolerance for floating point)
- [ ] `from thermite.ensemble import RandomForestClassifier` works  full sklearn import path compatibility
- [ ] Pipeline chains transformers and estimators correctly
- [ ] cross_val_score and GridSearchCV work end-to-end

### Performance
- [ ] Benchmark suite runs successfully and produces comparison tables
- [ ] At least 2x speedup over scikit-learn on RandomForest fit (100K rows, 20 features)
- [ ] At least 2x speedup over scikit-learn on GradientBoosting fit (100K rows, 20 features)
- [ ] At least 2x speedup over scikit-learn on KMeans fit (100K rows, 20 features)
- [ ] At least 2x speedup over scikit-learn on PCA fit_transform (100K rows, 50 features)

### Build & Install
- [ ] `pip install -e .` succeeds and makes `import thermite` work
- [ ] `cargo test` passes all Rust unit tests
- [ ] `pytest` passes all Python integration tests
- [ ] GitHub Actions CI workflow passes

### Documentation & Project Files
- [ ] README.md includes accurate installation instructions, benchmark results table, and usage examples
- [ ] API reference docs are generated and browsable
- [ ] DEVLOG.md is updated with a detailed work log entry documenting the build
