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

### The Numbers

| Operation | scikit-learn | Thermite | Speedup |
|-----------|-------------|---------|---------|
| LogisticRegression.fit (Small) | 0.0039s | 0.0003s | **12.58x** |
| KMeans.fit (Small) | 0.0367s | 0.0064s | **5.77x** |
| LinearRegression.fit (Medium) | 0.0162s | 0.0056s | **2.88x** |
| DecisionTree.fit (Large) | 4.0438s | 1.5063s | **2.68x** |
| RandomForest.fit (Small) | 0.0827s | 0.0180s | **4.60x** |

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

# Same API you already know
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
clf.fit(X_train, y_train)
print(f"Accuracy: {clf.score(X_test, y_test):.4f}")
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
2. **PyO3 bindings** (`python/thermite`): Thin Python layer that accepts numpy arrays, calls Rust, returns numpy arrays
3. **API compatibility**: Same `fit()`, `predict()`, `transform()`, `fit_transform()` interface as scikit-learn
4. **Zero-copy where possible**: Numpy arrays passed directly to Rust without copying via `numpy` PyO3 bindings

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
