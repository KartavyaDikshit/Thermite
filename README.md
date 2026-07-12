# Thermite ML

**A blazing-fast, Rust-accelerated machine learning library for Python — drop-in compatible with scikit-learn.**

> *Thermite: an exothermic reaction that burns at 2500°C. Your ML training should be just as fast.*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Rust](https://img.shields.io/badge/rust-1.75+-orange.svg)](https://www.rust-lang.org/)
[![PyPI](https://img.shields.io/badge/PyPI-v1.0.0-blue)](https://pypi.org/project/thermite-ml/)

---

##  Why Thermite?

scikit-learn is the most widely-used ML library in the world (40M+ monthly downloads), but its internals are built on NumPy/SciPy/Cython — fast for 2010, but bottlenecked by 2026 standards. 

**Thermite** rewrites the compute-heavy core natively in Rust using explicit SIMD, Rayon multithreading, and `matrixmultiply` optimizations. We expose the exact same Python API you already know. 
No new syntax. No migration guide. Just `import thermite` instead of `import sklearn`.

### Unmatched Capabilities
- **Zero-Copy Polars Integration:** Feed Apache Arrow `polars` DataFrames directly into Thermite's Rust core without any conversion or data duplication. 
- **GPU Acceleration (wgpu):** Native WebGPU/CUDA hardware acceleration backend `thermite-gpu`. Dispatch compute shaders for massive ensemble aggregations and matrix multiplications instantly by simply adding `device='gpu'`.
- **True Parallelism (No GIL):** Unlike Scikit-Learn which relies on heavy multiprocess pickling via `joblib`, Thermite releases the Python GIL during heavy computation. `GridSearchCV` and `RandomizedSearchCV` effortlessly scale across all your cores with zero IPC overhead.
- **Native Categorical & Sparse Support:** Decision Trees handle categorical features natively (bypassing One-Hot Encoding overhead). `LinearRegression` and `KMeans` natively ingest and optimize `scipy.sparse` matrices.
- **Out-of-Core Learning:** Memory too small? Use `.partial_fit()` to train `GaussianNB` or `LogisticRegression` on streaming datasets incrementally.

---

##  The Numbers (Performance Superiority)

Thermite offers incredible performance boosts while maintaining 100% accuracy parity.

| Operation | scikit-learn | Thermite | Speedup |
|-----------|-------------|---------|---------|
| `LogisticRegression.fit` (Sparse NLP) | 0.1068s | 0.0059s | **18.22x** |
| `LinearSVC.fit` (Sparse TF-IDF) | 0.0244s | 0.0022s | **10.99x** |
| `RandomForest.fit` (Categorical Splits) | 0.1806s | 0.0653s | **2.76x** |
| `LinearRegression.fit` (Dense 10k) | 0.0238s | 0.0100s | **2.37x** |
| `KMeans.fit` (Dense) | 0.0829s | 0.0356s | **2.33x** |
| `GridSearchCV` (100 folds, 8 cores) | ~14.0s | ~1.5s | **~9.3x** |

> *Tested on an M2 Apple Silicon chip. See `BENCHMARKS.md` for reproducible scripts.*

---

##  Installation

Thermite v1.0.0 is distributed as pre-compiled wheels for macOS, Linux, and Windows. No Rust toolchain required!

```bash
pip install thermite-ml
```

---

##  Quick Start: scikit-learn Drop-In

```python
# The API is 100% identical to scikit-learn
from thermite.ensemble import RandomForestClassifier
from thermite.model_selection import train_test_split
from thermite.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Opt-in to Hardware Acceleration with `device='gpu'`
clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, device='gpu')
clf.fit(X_train, y_train)

print(f"Accuracy: {clf.score(X_test, y_test):.4f}")
```

---

##  Zero-Copy Polars Integration

Traditional scikit-learn forces you to convert `polars` DataFrames to `pandas` or `numpy`, triggering a massive memory copy. Thermite natively ingests the underlying Apache Arrow memory buffers:

```python
import polars as pl
from thermite.linear_model import LogisticRegression
from thermite.polars_compat import make_polars_pipeline

df = pl.read_csv("100GB_dataset.csv")

# Instantly train directly on the Polars DataFrame
model = make_polars_pipeline(LogisticRegression())
model.fit(df.select(pl.exclude("target")), df["target"])
```

---

##  System Architecture

Thermite is structured to provide safety, performance, and portability:

1. **`thermite-core` (Rust):** The backbone. Implements the mathematical optimization routines using `ndarray` and `rayon`. Safe, memory-efficient, and brutally fast.
2. **`thermite-gpu` (Rust):** WebGPU (`wgpu`) based compute shader dispatch system targeting Vulkan, Metal, and DX12 dynamically.
3. **`thermite-binding` (Rust/PyO3):** The translation layer bridging Python NumPy arrays to Rust contiguous views with zero allocation.
4. **`thermite` (Python):** The high-level Python wrappers that mimic `scikit-learn` estimator APIs, implementing input validation and `BaseEstimator` compatibility.

---

##  Supported Algorithms

- **Linear Models:** `LinearRegression`, `Ridge`, `Lasso`, `LogisticRegression` (Binary & Multinomial OvR), `LinearSVC`.
- **Ensembles:** `RandomForestClassifier`, `RandomForestRegressor`, `GradientBoostingClassifier`, `GradientBoostingRegressor`.
- **Trees:** `DecisionTreeClassifier`, `DecisionTreeRegressor`.
- **Clustering:** `KMeans`, `DBSCAN`.
- **Decomposition:** `PCA`.
- **Probabilistic:** `GaussianNB`.
- **Preprocessing:** `StandardScaler`, `MinMaxScaler`, `OneHotEncoder`, `LabelEncoder`.
- **Pipelines:** `Pipeline`, `GridSearchCV`, `cross_val_score`.
- **Deep Learning Connectivity:** Native `__dlpack__` integrations for PyTorch/JAX `from_dlpack` zero-copy tensor passing.
- **AutoML:** Rust native `BayesianOptimizer` utilizing `Ridge` surrogates.

---

##  What Sets Thermite Apart & Competitive Comparison

While scikit-learn dominates the ML ecosystem (with over 100+ algorithms and extensive preprocessing), **Thermite** differentiates itself as an extreme performance overlay for production deployments.

**Why people still use scikit-learn:**
1. **Algorithm Breadth:** scikit-learn offers extensive specialized algorithms (e.g. Gaussian Mixture Models, complex Imputers like MICE) not yet ported to Thermite.
2. **Ecosystem Tooling:** Vastly wider array of third-party plugins.

**Why Thermite is unique:**
1. **Rust-Native & Zero-Copy:** While similar projects like `Intel(R) Extension for Scikit-learn` try to accelerate operations by monkey-patching Cython with daal4py, Thermite is rewritten ground-up in Rust. We achieve true zero-copy data transmission for Apache Arrow/Polars AND Deep Learning frameworks (PyTorch/JAX via `DLPack`).
2. **Distributed Computing Preparedness:** Thermite's Rust estimators derive `Serde` enabling high-speed `bincode` serialization out-of-the-box. This natively plugs into distributed execution engines like `Ray` and `Dask` without the heavy overhead of Python's standard `pickle`.
3. **Rust AutoML:** Instead of looping cross-validation in Python, Thermite provides a fast native `BayesianOptimizer`.
4. **Scale-up Gaps:** Future gaps to address for mass adoption include complex missing data strategies (`IterativeImputer`), adding more robust tree ensemble variants (like Histogram-based GBDT), and comprehensive categorical missing value handling.

