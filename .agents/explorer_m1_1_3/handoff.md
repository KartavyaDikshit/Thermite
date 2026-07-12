# Handoff Report: Build Setup and Packaging (Milestone M1-1)

This report details the recommended build setup, Cargo workspace configuration, Maturin/PyO3 Python packaging (`pyproject.toml`), and Python package directory structure for the Thermite project.

---

## 1. Observation

Direct observations from the workspace files and structure:
- **Project Structure**: Based on `list_dir` on the project root, no `Cargo.toml`, `pyproject.toml`, or source directories exist yet. The root contains:
  ```json
  {"name":".agents", "isDir":true}
  {"name":".git", "isDir":true}
  {"name":".gitignore", "sizeBytes":"154"}
  {"name":"DEVLOG.md", "sizeBytes":"3717"}
  {"name":"PROJECT.md", "sizeBytes":"4203"}
  {"name":"README.md", "sizeBytes":"3908"}
  ```
- **Architectural Details**: As defined in `/Users/kartavyadikshit/Projects/Thermite/PROJECT.md`, the layout must contain:
  - Root `Cargo.toml` and `pyproject.toml`
  - `crates/thermite-core/` (ML algorithms)
  - `crates/thermite-binding/` (PyO3 bindings crate)
  - `thermite/` (Python API package)
- **Interface Contracts**: As defined in `PROJECT.md`, the PyO3 bindings compile into a private binary module (e.g. `thermite._core`), accepting NumPy arrays (using `numpy-rust` or similar) and returning NumPy arrays.

---

## 2. Logic Chain

Based on these observations, the build setup and packaging files must satisfy the following:
1. **Workspace and Dependencies Management**:
   - A multi-crate Cargo workspace is required to isolate core Rust algorithms (`thermite-core`) from Python-specific binding boilerplate (`thermite-binding`).
   - Pinning versions in a central `[workspace.dependencies]` block in the root `Cargo.toml` ensures consistency, simplifies updates, and avoids mismatched versions of crates like `ndarray`, `rayon`, and `pyo3` between workspace members.
2. **PyO3 and Maturin Integration**:
   - To make `thermite` pip-installable in development mode via `pip install -e .` or built as wheels using `maturin`, a root `pyproject.toml` utilizing Maturin as the build backend is necessary.
   - Using PyO3 version `0.21.2` and NumPy crate `0.21.0` enables compatibility with modern PyO3 features, specifically utilizing the safe and explicit `Bound<'py, ...>` API pattern.
   - Setting `crate-type = ["cdylib"]` for `thermite-binding` and `name = "_core"` compiles the bindings crate into a shared library file (e.g. `_core.so` or `_core.pyd`) which Maturin can place inside the `thermite/` Python directory.
   - Activating the `abi3-py38` feature of PyO3 ensures stable ABI support, compiling single binary wheels compatible across multiple Python versions (3.8+).
3. **Python Package Scaffolding**:
   - The Python code should act as a clean wrapper around the private `_core` module, maintaining complete API compatibility with `scikit-learn` signature-wise, raising Python exceptions where appropriate, and validating inputs.
   - A stubbed layout of the Python package directory `thermite/` is needed to prepare for the development of subsequent preprocessing scalers/encoders (Milestone M1-3) and train_test_split (Milestone M1-2).

---

## 3. Caveats

- **Version Pinning**: The recommended crate versions (e.g. `pyo3` 0.21.2 and `numpy` 0.21.0) are fully compatible with each other. If downstream implementers modify these version pins, they must verify version compatibility matrices between `pyo3` and the `numpy` Rust library.
- **Stable ABI Limitations**: Building with `abi3-py38` restricts certain features of PyO3 to the stable C API. This is standard and sufficient for linear algebra and NumPy operations, but should be kept in mind if highly custom C-API extensions are proposed later.
- **Python Package Namespaces**: The private compiled binary is placed at `thermite._core` to align with modern conventions (like `pydantic_core`). If subsequent modules require other backend binaries, they should similarly be registered inside `crates/thermite-binding`.

---

## 4. Conclusion

We recommend creating the following configurations and directories inside the project workspace:

### 4.1. Root `Cargo.toml`
Create a Cargo workspace configuration defining the workspace members and centralizing core dependencies.
```toml
# /Users/kartavyadikshit/Projects/Thermite/Cargo.toml
[workspace]
resolver = "2"
members = [
    "crates/thermite-core",
    "crates/thermite-binding",
]

[workspace.dependencies]
ndarray = { version = "0.15.6", features = ["rayon"] }
rayon = "1.8.1"
rand = "0.8.5"
pyo3 = { version = "0.21.2", features = ["extension-module", "abi3", "abi3-py38"] }
numpy = "0.21.0"
thiserror = "1.0.58"
serde = { version = "1.0.197", features = ["derive"] }
```

### 4.2. Root `pyproject.toml`
Configure Maturin as the build backend, define package metadata, and set up python package options.
```toml
# /Users/kartavyadikshit/Projects/Thermite/pyproject.toml
[build-system]
requires = ["maturin>=1.5,<2.0"]
build-backend = "maturin"

[project]
name = "thermite"
version = "0.1.0"
description = "A Rust-accelerated, scikit-learn-compatible machine learning library for Python"
readme = "README.md"
requires-python = ">=3.8"
license = { text = "MIT" }
authors = [
    { name = "Kartavya Dikshit" }
]
classifiers = [
    "Programming Language :: Rust",
    "Programming Language :: Python :: Implementation :: CPython",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
dependencies = [
    "numpy>=1.20.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "scikit-learn>=1.0.0",
    "ruff>=0.1.0",
]

[tool.maturin]
manifest-path = "crates/thermite-binding/Cargo.toml"
python-source = "."
module-name = "thermite._core"
features = ["pyo3/abi3-py38"]
```

### 4.3. Crate `thermite-core` Configuration and Initial Src
Create the workspace member that houses the core pure-Rust implementation.

**`crates/thermite-core/Cargo.toml`**:
```toml
# /Users/kartavyadikshit/Projects/Thermite/crates/thermite-core/Cargo.toml
[package]
name = "thermite-core"
version = "0.1.0"
edition = "2021"

[lib]
name = "thermite_core"
crate-type = ["rlib"]

[dependencies]
ndarray = { workspace = true }
rayon = { workspace = true }
rand = { workspace = true }
thiserror = { workspace = true }
serde = { workspace = true }
```

**`crates/thermite-core/src/lib.rs`**:
```rust
// /Users/kartavyadikshit/Projects/Thermite/crates/thermite-core/src/lib.rs
pub mod preprocessing;
pub mod model_selection;
```

**`crates/thermite-core/src/preprocessing.rs`**:
```rust
// /Users/kartavyadikshit/Projects/Thermite/crates/thermite-core/src/preprocessing.rs
use ndarray::{Array1, Array2, Axis};

#[derive(Debug, Clone)]
pub struct StandardScaler {
    pub with_mean: bool,
    pub with_std: bool,
    pub mean: Option<Array1<f64>>,
    pub var: Option<Array1<f64>>,
    pub scale: Option<Array1<f64>>,
    pub n_samples_seen: usize,
}

impl StandardScaler {
    pub fn new(with_mean: bool, with_std: bool) -> Self {
        Self {
            with_mean,
            with_std,
            mean: None,
            var: None,
            scale: None,
            n_samples_seen: 0,
        }
    }

    pub fn fit(&mut self, x: &Array2<f64>) {
        let n_samples = x.nrows();
        if n_samples == 0 {
            return;
        }

        if self.with_mean {
            let mean = x.mean_axis(Axis(0)).unwrap();
            self.mean = Some(mean);
        }

        if self.with_std {
            let var = x.var_axis(Axis(0), 0.0);
            let mut scale = var.mapv(|v| v.sqrt());
            // Prevent division by zero
            scale.mapv_inplace(|s| if s == 0.0 { 1.0 } else { s });
            self.var = Some(var);
            self.scale = Some(scale);
        }

        self.n_samples_seen += n_samples;
    }

    pub fn transform(&self, x: &Array2<f64>) -> Array2<f64> {
        let mut res = x.clone();
        if let Some(ref mean) = self.mean {
            res = res - mean;
        }
        if let Some(ref scale) = self.scale {
            res = res / scale;
        }
        res
    }
}
```

**`crates/thermite-core/src/model_selection.rs`**:
```rust
// /Users/kartavyadikshit/Projects/Thermite/crates/thermite-core/src/model_selection.rs
// Placeholder for train_test_split utility
```

### 4.4. Crate `thermite-binding` Configuration and Initial Src
Create the workspace member that wraps `thermite-core` using PyO3 and generates Python-callable constructs.

**`crates/thermite-binding/Cargo.toml`**:
```toml
# /Users/kartavyadikshit/Projects/Thermite/crates/thermite-binding/Cargo.toml
[package]
name = "thermite-binding"
version = "0.1.0"
edition = "2021"

[lib]
name = "_core"
crate-type = ["cdylib"]

[dependencies]
thermite-core = { path = "../thermite-core" }
pyo3 = { workspace = true, features = ["abi3-py38"] }
numpy = { workspace = true }
ndarray = { workspace = true }
```

**`crates/thermite-binding/src/lib.rs`**:
```rust
// /Users/kartavyadikshit/Projects/Thermite/crates/thermite-binding/src/lib.rs
use pyo3::prelude::*;
use numpy::{PyReadonlyArray2, PyArray2, IntoPyArray};
use thermite_core::preprocessing::StandardScaler as CoreStandardScaler;

#[pyclass(name = "PyStandardScaler")]
pub struct PyStandardScaler {
    core: CoreStandardScaler,
}

#[pymethods]
impl PyStandardScaler {
    #[new]
    #[pyo3(signature = (with_mean=true, with_std=true))]
    fn new(with_mean: bool, with_std: bool) -> Self {
        PyStandardScaler {
            core: CoreStandardScaler::new(with_mean, with_std),
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<f64>) {
        let x_arr = x.as_array();
        self.core.fit(&x_arr.to_owned());
    }

    fn transform<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<f64>) -> Bound<'py, PyArray2<f64>> {
        let x_arr = x.as_array();
        let res = self.core.transform(&x_arr.to_owned());
        res.into_pyarray_bound(py)
    }

    #[getter]
    fn mean<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray2<f64>>>> {
        if let Some(ref mean) = self.core.mean {
            Ok(Some(mean.clone().into_pyarray_bound(py)))
        } else {
            Ok(None)
        }
    }

    #[getter]
    fn var<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray2<f64>>>> {
        if let Some(ref var) = self.core.var {
            Ok(Some(var.clone().into_pyarray_bound(py)))
        } else {
            Ok(None)
        }
    }

    #[getter]
    fn scale<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray2<f64>>>> {
        if let Some(ref scale) = self.core.scale {
            Ok(Some(scale.clone().into_pyarray_bound(py)))
        } else {
            Ok(None)
        }
    }

    #[getter]
    fn n_samples_seen(&self) -> usize {
        self.core.n_samples_seen
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyStandardScaler>()?;
    Ok(())
}
```

### 4.5. Python Package Structure (`thermite/`)
Create the package directory containing Python wrapper classes and modules.

**`thermite/__init__.py`**:
```python
# /Users/kartavyadikshit/Projects/Thermite/thermite/__init__.py
"""Thermite: A Rust-accelerated, scikit-learn-compatible machine learning library."""

from . import preprocessing
from . import model_selection

__version__ = "0.1.0"
__all__ = [
    "preprocessing",
    "model_selection",
]
```

**`thermite/preprocessing.py`**:
```python
# /Users/kartavyadikshit/Projects/Thermite/thermite/preprocessing.py
import numpy as np
from thermite._core import PyStandardScaler

class StandardScaler:
    """Standardize features by removing the mean and scaling to unit variance."""
    
    def __init__(self, *, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std
        self._backend = PyStandardScaler(with_mean=with_mean, with_std=with_std)
        self.mean_ = None
        self.var_ = None
        self.scale_ = None
        self.n_samples_seen_ = 0

    def fit(self, X, y=None):
        """Compute the mean and std to be used for later scaling."""
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X_arr.ndim}D array instead")
        if X_arr.shape[0] == 0:
            raise ValueError("Empty training data provided")
            
        self._backend.fit(X_arr)
        self.mean_ = self._backend.mean
        self.var_ = self._backend.var
        self.scale_ = self._backend.scale
        self.n_samples_seen_ = self._backend.n_samples_seen
        return self

    def transform(self, X, copy=None):
        """Perform standardization by centering and scaling."""
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X_arr.ndim}D array instead")
            
        return self._backend.transform(X_arr)

    def fit_transform(self, X, y=None):
        """Fit to data, then transform it."""
        return self.fit(X, y).transform(X)


class MinMaxScaler:
    """Transform features by scaling each feature to a given range (placeholder)."""
    def __init__(self, feature_range=(0, 1), *, copy=True, clip=False):
        self.feature_range = feature_range
        self.copy = copy
        self.clip = clip

    def fit(self, X, y=None):
        raise NotImplementedError("MinMaxScaler will be implemented in M1-3")

    def transform(self, X):
        raise NotImplementedError("MinMaxScaler will be implemented in M1-3")

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class LabelEncoder:
    """Encode target labels with value between 0 and n_classes-1 (placeholder)."""
    def fit(self, y):
        raise NotImplementedError("LabelEncoder will be implemented in M1-3")

    def transform(self, y):
        raise NotImplementedError("LabelEncoder will be implemented in M1-3")

    def fit_transform(self, y):
        return self.fit(y).transform(y)


class OneHotEncoder:
    """Encode categorical features as a one-hot numeric array (placeholder)."""
    def __init__(self, *, categories='auto', drop=None, sparse_output=True, dtype=np.float64, handle_unknown='error'):
        self.categories = categories
        self.drop = drop
        self.sparse_output = sparse_output
        self.dtype = dtype
        self.handle_unknown = handle_unknown

    def fit(self, X, y=None):
        raise NotImplementedError("OneHotEncoder will be implemented in M1-3")

    def transform(self, X):
        raise NotImplementedError("OneHotEncoder will be implemented in M1-3")

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
```

**`thermite/model_selection.py`**:
```python
# /Users/kartavyadikshit/Projects/Thermite/thermite/model_selection.py
import numpy as np

def train_test_split(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None):
    """Split arrays or matrices into random train and test subsets (placeholder)."""
    raise NotImplementedError("train_test_split will be implemented in M1-2")
```

**`thermite/tree.py`**:
```python
# /Users/kartavyadikshit/Projects/Thermite/thermite/tree.py
# Placeholder for DecisionTreeClassifier and DecisionTreeRegressor (Milestone 3)
```

**`thermite/ensemble.py`**:
```python
# /Users/kartavyadikshit/Projects/Thermite/thermite/ensemble.py
# Placeholder for RandomForestClassifier, RandomForestRegressor, etc. (Milestone 3)
```

**`thermite/cluster.py`**:
```python
# /Users/kartavyadikshit/Projects/Thermite/thermite/cluster.py
# Placeholder for KMeans and DBSCAN (Milestone 3)
```

**`thermite/decomposition.py`**:
```python
# /Users/kartavyadikshit/Projects/Thermite/thermite/decomposition.py
# Placeholder for PCA (Milestone 3)
```

**`thermite/metrics.py`**:
```python
# /Users/kartavyadikshit/Projects/Thermite/thermite/metrics.py
# Placeholder for performance metrics (Milestone 2)
```

**`thermite/pipeline.py`**:
```python
# /Users/kartavyadikshit/Projects/Thermite/thermite/pipeline.py
# Placeholder for Pipeline (Milestone 4)
```

---

## 5. Verification Method

To verify this configuration setup independently once the files are written by the implementer agent:

### 5.1. Compilation Verification
Run from the root of the project:
```bash
cargo build --all-targets
```
*Expected Output*: Compile logs showing successful build of `thermite-core` followed by `thermite-binding` with no warnings or errors.

### 5.2. Installation Verification
Inside a Python virtual environment at the project root:
```bash
pip install -e .[dev]
```
*Expected Output*: Installs numpy, pytest, scikit-learn, and compiles the Rust package in editable development mode, putting `_core.so` (or `_core.pyd`) under the `thermite/` directory.

### 5.3. Basic End-to-End Verification
Create a simple python test script (e.g. `test_setup.py`) or run interactively:
```python
import numpy as np
from thermite.preprocessing import StandardScaler

X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
scaler = StandardScaler()
scaler.fit(X)

print("Mean:", scaler.mean_)
print("Scale:", scaler.scale_)
print("Transformed:\n", scaler.transform(X))
```
*Expected Output*:
```text
Mean: [3. 4.]
Scale: [1.63299316 1.63299316]
Transformed:
 [[-1.22474487 -1.22474487]
  [ 0.          0.        ]
  [ 1.22474487  1.22474487]]
```

### 5.4. Invalidation Conditions
The verification fails if:
- `cargo build` fails due to package version mismatches.
- `pip install -e .` fails due to compilation errors in Maturin or PyO3 configuration issues.
- `import thermite._core` raises `ModuleNotFoundError` or `ImportError` due to mismatched target or path layout in `pyproject.toml`.
