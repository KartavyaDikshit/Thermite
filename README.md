#  Thermite

**A blazing-fast, Rust-accelerated machine learning library for Python  drop-in compatible with scikit-learn.**

> *Thermite: an exothermic reaction that burns at 2500C. Your ML training should be just as fast.*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Rust](https://img.shields.io/badge/rust-1.75+-orange.svg)](https://www.rust-lang.org/)

---

## Why Thermite?

scikit-learn is the most widely-used ML library in the world (40M+ monthly downloads), but its internals are built on NumPy/SciPy/Cython  fast for 2010, slow by 2026 standards.

**Thermite** rewrites the compute-heavy core in Rust and exposes the same Python API you already know. No new syntax. No migration guide. Just `import thermite` instead of `import sklearn`.

### Supported Models & Tools (Phase 1 & 2 Complete, Phase 3 In Progress)
- **Linear Models**: LinearRegression, Ridge, Lasso, LogisticRegression (Binary & Multiclass OvR, Native Sparse CSR, `partial_fit` streaming), LinearSVC (Binary & Multiclass OvR, Native Sparse CSR)
- **Tree & Ensemble Models**: DecisionTreeClassifier, DecisionTreeRegressor, RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor (All support Native Categorical Splits, avoiding memory-heavy one-hot encoding)
- **Clustering**: KMeans (Native Sparse CSR support)
- **Decomposition**: PCA (Subspace iteration optimized)
- **Probabilistic Models**: GaussianNB (Includes `partial_fit` for Out-of-Core online learning)
- **Pipelines & Tuning**: Fully compliant `Pipeline` and `ColumnTransformer`. `GridSearchCV` and `RandomizedSearchCV` with true multi-core parallelism via GIL release (zero pickling)
- **Hardware Acceleration**: `thermite-gpu` crate with wgpu/CUDA dispatch. Select via `device='cuda'`. CPU fallback always available.

### The Numbers

| Operation | scikit-learn | Thermite | Speedup |
|-----------|-------------|---------|---------|
| LinearRegression.fit (Dense) | 0.0238s | 0.0100s | **2.37x** |
| KMeans.fit (Dense) | 0.0829s | 0.0356s | **2.33x** |
| LogisticRegression.fit (Sparse) | 0.1068s | 0.0059s | **18.22x** |
| LinearSVC.fit (Sparse) | 0.0244s | 0.0022s | **10.99x** |
| DecisionTree.fit (Categorical) | 0.1405s | 0.0564s | **2.49x** |
| RandomForest.fit (Categorical) | 0.1806s | 0.0653s | **2.76x** |
| GaussianNB.fit (Dense) | 0.0066s | 0.0039s | **1.69x** |

> See `BENCHMARKS.md` for full detailed performance comparisons across varying datasets.

## Installation

```bash
pip install thermite-ml
```

## Quick Start

```python
# Drop-in replacement for scikit-learn
from thermite.ensemble import RandomForestClassifier
from thermite.model_selection import train_test_split
from thermite.preprocessing import StandardScaler

# Same API you already know, but faster
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Hardware acceleration with `device='cuda'` or `device='gpu'`
clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, device='gpu')
clf.fit(X_train, y_train)
print(f"Accuracy: {clf.score(X_test, y_test):.4f}")
```

### Zero-copy Polars Integration

Thermite natively supports `polars` DataFrames via Apache Arrow memory formats without copying data (where `dtype` allows):

```python
import polars as pl
from thermite.linear_model import LogisticRegression
from thermite.polars_compat import make_polars_pipeline, from_polars

df = pl.read_csv("large_dataset.csv")

# Extract features and target without copying
X, y = from_polars(df, target_col="target")

# Or wrap the model to ingest DataFrames directly
model = make_polars_pipeline(LogisticRegression())
model.fit(df.select(pl.exclude("target")), df["target"])
```

## Architecture

```
thermite/
 crates/
    thermite-core/        # Rust ML algorithms
    thermite-linalg/      # Linear algebra primitives
    thermite-io/          # Fast data loading/serialization
 python/
    thermite/             # Python API (PyO3)
        linear_model/     # Linear/Logistic Regression, Ridge, Lasso
        tree/             # Decision Trees
        ensemble/         # Random Forest, Gradient Boosting
        cluster/          # KMeans, DBSCAN
        preprocessing/    # Scalers, Encoders, Imputers
        model_selection/  # Cross-validation, Grid Search
        decomposition/    # PCA, SVD
        metrics/          # Scoring functions
 benchmarks/               # Head-to-head vs scikit-learn
 docs/                     # Documentation
```

## How It Works

1. **Rust core** (`thermite-core`): All ML algorithms implemented in pure Rust with Rayon for automatic parallelism
2. **GPU Backend** (`thermite-gpu`): Hardware acceleration for massive matrices using wgpu/WGSL compute shaders
3. **PyO3 bindings** (`thermite-binding`): Thin Python layer that bridges Numpy/Polars to Rust zero-copy
4. **API compatibility**: Same `fit()`, `predict()`, `transform()`, `partial_fit()` interface as scikit-learn

## Built With

- [Rust](https://www.rust-lang.org/)  Systems programming language
- [PyO3](https://github.com/PyO3/pyo3)  Rust  Python interop
- [maturin](https://github.com/PyO3/maturin)  Build and publish Rust-backed Python packages
- [Rayon](https://github.com/rayon-rs/rayon)  Data parallelism for Rust
- [ndarray](https://github.com/rust-ndarray/ndarray)  N-dimensional arrays for Rust

## Contributing

Thermite is in active development. Contributions welcome  see [DEVLOG.md](DEVLOG.md) for the current roadmap.

## License

MIT
