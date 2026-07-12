# Handoff Report - Build Setup & Packaging (Milestone M1-1)

## 1. Observation

We directly observed that the repository `/Users/kartavyadikshit/Projects/Thermite/` is currently in a pre-code state. Running file discovery showed only the following metadata and documentation files:
- `.gitignore` (contains typical build/virtualenv rules)
- `DEVLOG.md`
- `PROJECT.md`
- `README.md`
- `.agents/` directory

From `/Users/kartavyadikshit/Projects/Thermite/PROJECT.md`:
> "Rust Core Workspace:
> - `crates/thermite-core`: Implements the ML algorithms (regression, classification, clustering, decomposition, preprocessing, etc.) using `ndarray` and parallelized via `Rayon`.
> - `crates/thermite-linalg` (optional/integrated)...
> - `crates/thermite-binding`: PyO3 bindings crate.
> Python Bindings & Package:
> - `thermite/`: The Python package directory containing wrappers that mimic scikit-learn module structure.
> - PyO3 bindings compiled into a private binary module (e.g., `thermite._core` or `thermite.binding`)."

From `/Users/kartavyadikshit/Projects/Thermite/.agents/sub_orch_m1/SCOPE.md`:
> "M1-1 | Build Setup & Packaging | Create `Cargo.toml`, `pyproject.toml`, and Python package structure. Verify maturin develop compiles and installs."

## 2. Logic Chain

1. **Cargo Workspace Configuration**:
   To group `thermite-core` and `thermite-binding`, we require a workspace-level `Cargo.toml` at the project root defining the member directories (`crates/thermite-core` and `crates/thermite-binding`) and resolving with version `2` of the feature resolver.

2. **Maturin and PyO3 Integration**:
   To compile Rust code into a Python-accessible binary extension module:
   - We need a root-level `pyproject.toml` with `maturin` as the build-system backend.
   - The `[tool.maturin]` metadata must point to the bindings manifest (`crates/thermite-binding/Cargo.toml`) and define the module name as `thermite._core` (with `python-source = "."` to keep the python wrappers directly at the root package directory `thermite/`).
   - The binding crate `thermite-binding` must configure its library target with `crate-type = ["cdylib"]` so it outputs a shared library importable by Python.

3. **Crate Decoupling**:
   - `thermite-core` contains pure Rust numerical code (using `ndarray`, `rayon`, and `rand` crates) and contains no PyO3 dependency. This allows writing lightweight, fast Rust tests (`cargo test`) and ensures the core ML algorithms can be reused/maintained separate from the Python interpreter layer.
   - `thermite-binding` contains the PyO3 glue layer. It depends on `pyo3` and `numpy` crates, translates NumPy inputs to Rust `ndarray` views, calls the underlying core structures, and converts outputs back to NumPy arrays.

4. **Python Package & API Compatibility**:
   - To mimic scikit-learn (e.g. `from thermite.preprocessing import StandardScaler`), we create `thermite/` at the root, containing `__init__.py`, `preprocessing.py`, and `model_selection.py`.
   - The Python classes (e.g. `StandardScaler`) wrap the compiled Rust classes (`_core.StandardScalerRust`), validating arguments (e.g., matching shapes), maintaining scikit-learn attribute naming conventions (e.g., `mean_`, `var_`, `n_samples_seen_`), and exposing `fit()`, `transform()`, and `fit_transform()`.
   - For `train_test_split`, passing high-dimensional Python arrays into Rust, allocating them, and copying them back is inefficient. A better architectural decision is to pass the array size into `_core.get_split_indices(...)`, let Rust handle shuffling and division of index lists via the `rand` crate (using reproducible seeds), return two index lists (`train_indices` and `test_indices`) to Python, and perform zero-copy fancy indexing in Python.

## 3. Caveats

- **Read-Only Restriction**: Because we are in read-only investigation mode, we did not write these files directly into the repository, nor did we execute the build and check steps. They must be created by the implementer.
- **Categorical Types in Rust**: In `LabelEncoder` and `OneHotEncoder`, Python passes arbitrary lists of strings or integers. For the Rust backend, casting inputs to `String` (i.e. categorical labels as strings) is recommended to avoid dealing with complex generic type matching across PyO3 boundaries in this phase.
- **NumPy Matrix Slicing**: Slicing arrays via `train_test_split` creates views or copies in numpy. Index-based splitting avoids copying values during the Rust-Python boundary transit, but NumPy still makes copies/views depending on layout.

## 4. Conclusion

We recommend creating the following 11 files with the structures detailed below.

### 4.1. Workspace Configuration

#### `Cargo.toml` (Root Workspace)
```toml
[workspace]
members = [
    "crates/thermite-core",
    "crates/thermite-binding",
]
resolver = "2"
```

#### `pyproject.toml` (Root Python Packaging)
```toml
[build-system]
requires = ["maturin>=1.5,<2.0"]
build-backend = "maturin"

[project]
name = "thermite-ml"
version = "0.1.0"
description = "A blazing-fast, Rust-accelerated machine learning library for Python — drop-in compatible with scikit-learn."
readme = "README.md"
requires-python = ">=3.9"
authors = [
    { name = "Kartavya Dikshit" }
]
classifiers = [
    "Programming Language :: Python :: 3",
    "Programming Language :: Rust",
    "Operating System :: OS Independent",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
dependencies = [
    "numpy>=1.20",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "scikit-learn>=1.0",
]

[tool.maturin]
features = ["pyo3/extension-module"]
manifest-path = "crates/thermite-binding/Cargo.toml"
module-name = "thermite._core"
python-source = "."
```

---

### 4.2. Core Library Crate (`thermite-core`)

#### `crates/thermite-core/Cargo.toml`
```toml
[package]
name = "thermite-core"
version = "0.1.0"
edition = "2021"

[dependencies]
ndarray = { version = "0.15", features = ["rayon"] }
rayon = "1.10"
rand = "0.8"
rand_chacha = "0.3"
thiserror = "1.0"
```

#### `crates/thermite-core/src/lib.rs`
```rust
pub mod preprocessing;
pub mod model_selection;
```

#### `crates/thermite-core/src/preprocessing.rs`
```rust
use ndarray::{Array1, Array2, ArrayView2, Axis};
use std::collections::HashSet;

/// StandardScaler scales data to have zero mean and unit variance.
pub struct StandardScaler {
    pub with_mean: bool,
    pub with_std: bool,
    pub mean: Option<Array1<f64>>,
    pub var: Option<Array1<f64>>,
    pub n_samples_seen: usize,
}

impl StandardScaler {
    pub fn new(with_mean: bool, with_std: bool) -> Self {
        Self {
            with_mean,
            with_std,
            mean: None,
            var: None,
            n_samples_seen: 0,
        }
    }

    pub fn fit(&mut self, x: ArrayView2<f64>) {
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
            self.var = Some(var);
        }

        self.n_samples_seen = n_samples;
    }

    pub fn transform(&self, mut x: Array2<f64>) -> Array2<f64> {
        if let Some(ref mean) = self.mean {
            x = x - mean;
        }
        if let Some(ref var) = self.var {
            let std = var.mapv(|v| if v == 0.0 { 1.0 } else { v.sqrt() });
            x = x / std;
        }
        x
    }
}

/// MinMaxScaler scales features to a user-defined range.
pub struct MinMaxScaler {
    pub min_val: f64,
    pub max_val: f64,
    pub data_min: Option<Array1<f64>>,
    pub data_max: Option<Array1<f64>>,
    pub data_range: Option<Array1<f64>>,
    pub scale: Option<Array1<f64>>,
    pub min: Option<Array1<f64>>,
    pub n_samples_seen: usize,
}

impl MinMaxScaler {
    pub fn new(min_val: f64, max_val: f64) -> Self {
        Self {
            min_val,
            max_val,
            data_min: None,
            data_max: None,
            data_range: None,
            scale: None,
            min: None,
            n_samples_seen: 0,
        }
    }

    pub fn fit(&mut self, x: ArrayView2<f64>) {
        let n_samples = x.nrows();
        if n_samples == 0 {
            return;
        }

        let mut data_min = Array1::zeros(x.ncols());
        let mut data_max = Array1::zeros(x.ncols());

        for col in 0..x.ncols() {
            let col_view = x.column(col);
            let mut min_val = col_view[0];
            let mut max_val = col_view[0];
            for &val in col_view.iter() {
                if val < min_val {
                    min_val = val;
                }
                if val > max_val {
                    max_val = val;
                }
            }
            data_min[col] = min_val;
            data_max[col] = max_val;
        }

        let data_range = &data_max - &data_min;
        let scale = data_range.mapv(|r| {
            if r == 0.0 {
                0.0
            } else {
                (self.max_val - self.min_val) / r
            }
        });
        let min = &self.min_val - (&data_min * &scale);

        self.data_min = Some(data_min);
        self.data_max = Some(data_max);
        self.data_range = Some(data_range);
        self.scale = Some(scale);
        self.min = Some(min);
        self.n_samples_seen = n_samples;
    }

    pub fn transform(&self, mut x: Array2<f64>) -> Array2<f64> {
        if let (Some(ref scale), Some(ref min)) = (&self.scale, &self.min) {
            x = x * scale + min;
        }
        x
    }
}

/// LabelEncoder encodes labels (as strings) to integer indices.
pub struct LabelEncoder {
    pub classes: Vec<String>,
}

impl LabelEncoder {
    pub fn new() -> Self {
        Self { classes: Vec::new() }
    }

    pub fn fit(&mut self, y: &[String]) {
        let mut set = HashSet::new();
        for label in y {
            set.insert(label.clone());
        }
        let mut classes: Vec<String> = set.into_iter().collect();
        classes.sort();
        self.classes = classes;
    }

    pub fn transform(&self, y: &[String]) -> Result<Vec<usize>, String> {
        y.iter()
            .map(|label| {
                self.classes
                    .binary_search(label)
                    .map_err(|_| format!("Label '{}' not seen during fit.", label))
            })
            .collect()
    }

    pub fn inverse_transform(&self, indices: &[usize]) -> Result<Vec<String>, String> {
        indices
            .iter()
            .map(|&idx| {
                if idx < self.classes.len() {
                    Ok(self.classes[idx].clone())
                } else {
                    Err(format!("Index {} is out of range for classes of size {}", idx, self.classes.len()))
                }
            })
            .collect()
    }
}

/// OneHotEncoder encodes categorical columns into binary columns.
pub struct OneHotEncoder {
    pub categories_spec: String,
    pub drop_spec: Option<String>,
    pub sparse_output: bool,
    pub categories: Vec<Vec<String>>,
}

impl OneHotEncoder {
    pub fn new(categories_spec: String, drop_spec: Option<String>, sparse_output: bool) -> Self {
        Self {
            categories_spec,
            drop_spec,
            sparse_output,
            categories: Vec::new(),
        }
    }

    pub fn fit(&mut self, x: ArrayView2<&str>) {
        let n_cols = x.ncols();
        let mut categories = Vec::with_capacity(n_cols);

        for col in 0..n_cols {
            let col_view = x.column(col);
            let mut unique_vals = HashSet::new();
            for &val in col_view.iter() {
                unique_vals.insert(val.to_string());
            }
            let mut cats: Vec<String> = unique_vals.into_iter().collect();
            cats.sort();
            categories.push(cats);
        }
        self.categories = categories;
    }

    pub fn transform(&self, x: ArrayView2<&str>) -> Array2<f64> {
        let n_samples = x.nrows();
        
        let mut total_out_cols = 0;
        for col_cats in &self.categories {
            total_out_cols += col_cats.len();
        }

        let mut output = Array2::zeros((n_samples, total_out_cols));

        for row in 0..n_samples {
            let mut current_col_offset = 0;
            for col in 0..x.ncols() {
                let val = x[[row, col]];
                let col_cats = &self.categories[col];
                if let Some(pos) = col_cats.iter().position(|c| c == val) {
                    output[[row, current_col_offset + pos]] = 1.0;
                }
                current_col_offset += col_cats.len();
            }
        }
        output
    }
}
```

#### `crates/thermite-core/src/model_selection.rs`
```rust
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

pub fn get_split_indices(
    n_samples: usize,
    test_size: Option<f64>,
    train_size: Option<f64>,
    shuffle: bool,
    seed: Option<u64>,
) -> (Vec<usize>, Vec<usize>) {
    let mut indices: Vec<usize> = (0..n_samples).collect();

    if shuffle {
        let mut rng = match seed {
            Some(s) => ChaCha8Rng::seed_from_u64(s),
            None => ChaCha8Rng::from_entropy(),
        };
        indices.shuffle(&mut rng);
    }

    let n_test = match (test_size, train_size) {
        (Some(ts), _) => {
            if ts >= 1.0 {
                ts as usize
            } else {
                (n_samples as f64 * ts).round() as usize
            }
        }
        (None, Some(tr)) => {
            let tr_count = if tr >= 1.0 {
                tr as usize
            } else {
                (n_samples as f64 * tr).round() as usize
            };
            n_samples.saturating_sub(tr_count)
        }
        (None, None) => {
            (n_samples as f64 * 0.25).round() as usize
        }
    };

    let n_test = std::cmp::min(n_test, n_samples);
    let n_train = n_samples - n_test;

    let train_indices = indices[0..n_train].to_vec();
    let test_indices = indices[n_train..].to_vec();

    (train_indices, test_indices)
}
```

---

### 4.3. PyO3 Binding Crate (`thermite-binding`)

#### `crates/thermite-binding/Cargo.toml`
```toml
[package]
name = "thermite-binding"
version = "0.1.0"
edition = "2021"

[lib]
name = "_core"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.21", features = ["extension-module"] }
numpy = "0.21"
thermite-core = { path = "../thermite-core" }
```

#### `crates/thermite-binding/src/lib.rs`
```rust
use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyReadonlyArray2, ToPyArray};
use thermite_core::preprocessing::{
    StandardScaler as CoreStandardScaler,
    MinMaxScaler as CoreMinMaxScaler,
    LabelEncoder as CoreLabelEncoder,
    OneHotEncoder as CoreOneHotEncoder,
};
use thermite_core::model_selection::get_split_indices as core_get_split_indices;

#[pyclass(name = "StandardScalerRust")]
pub struct PyStandardScaler {
    inner: CoreStandardScaler,
}

#[pymethods]
impl PyStandardScaler {
    #[new]
    #[pyo3(signature = (with_mean=true, with_std=true))]
    fn new(with_mean: bool, with_std: bool) -> Self {
        Self {
            inner: CoreStandardScaler::new(with_mean, with_std),
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<f64>) {
        self.inner.fit(x.as_array());
    }

    fn transform<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<f64>) -> Bound<'py, PyArray2<f64>> {
        let x_owned = x.to_owned_array();
        let result = self.inner.transform(x_owned);
        result.to_pyarray_bound(py)
    }

    #[getter]
    fn mean<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner.mean.as_ref().map(|m| m.to_pyarray_bound(py))
    }

    #[getter]
    fn var<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner.var.as_ref().map(|v| v.to_pyarray_bound(py))
    }

    #[getter]
    fn n_samples_seen(&self) -> usize {
        self.inner.n_samples_seen
    }
}

#[pyclass(name = "MinMaxScalerRust")]
pub struct PyMinMaxScaler {
    inner: CoreMinMaxScaler,
}

#[pymethods]
impl PyMinMaxScaler {
    #[new]
    #[pyo3(signature = (min_val=0.0, max_val=1.0))]
    fn new(min_val: f64, max_val: f64) -> Self {
        Self {
            inner: CoreMinMaxScaler::new(min_val, max_val),
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<f64>) {
        self.inner.fit(x.as_array());
    }

    fn transform<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<f64>) -> Bound<'py, PyArray2<f64>> {
        let x_owned = x.to_owned_array();
        let result = self.inner.transform(x_owned);
        result.to_pyarray_bound(py)
    }

    #[getter]
    fn data_min<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner.data_min.as_ref().map(|m| m.to_pyarray_bound(py))
    }

    #[getter]
    fn data_max<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner.data_max.as_ref().map(|m| m.to_pyarray_bound(py))
    }

    #[getter]
    fn data_range<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner.data_range.as_ref().map(|m| m.to_pyarray_bound(py))
    }

    #[getter]
    fn scale<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner.scale.as_ref().map(|m| m.to_pyarray_bound(py))
    }

    #[getter]
    fn min<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner.min.as_ref().map(|m| m.to_pyarray_bound(py))
    }

    #[getter]
    fn n_samples_seen(&self) -> usize {
        self.inner.n_samples_seen
    }
}

#[pyclass(name = "LabelEncoderRust")]
pub struct PyLabelEncoder {
    inner: CoreLabelEncoder,
}

#[pymethods]
impl PyLabelEncoder {
    #[new]
    fn new() -> Self {
        Self {
            inner: CoreLabelEncoder::new(),
        }
    }

    fn fit(&mut self, y: Vec<String>) {
        self.inner.fit(&y);
    }

    fn transform(&self, y: Vec<String>) -> PyResult<Vec<usize>> {
        self.inner.transform(&y).map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
    }

    fn inverse_transform(&self, indices: Vec<usize>) -> PyResult<Vec<String>> {
        self.inner.inverse_transform(&indices).map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
    }

    fn classes(&self) -> Vec<String> {
        self.inner.classes.clone()
    }
}

#[pyclass(name = "OneHotEncoderRust")]
pub struct PyOneHotEncoder {
    inner: CoreOneHotEncoder,
}

#[pymethods]
impl PyOneHotEncoder {
    #[new]
    #[pyo3(signature = (categories_spec=String::from("auto"), drop_spec=None, sparse_output=false))]
    fn new(categories_spec: String, drop_spec: Option<String>, sparse_output: bool) -> Self {
        Self {
            inner: CoreOneHotEncoder::new(categories_spec, drop_spec, sparse_output),
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<&str>) {
        self.inner.fit(x.as_array());
    }

    fn transform<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<&str>) -> Bound<'py, PyArray2<f64>> {
        let result = self.inner.transform(x.as_array());
        result.to_pyarray_bound(py)
    }

    fn categories(&self) -> Vec<Vec<String>> {
        self.inner.categories.clone()
    }
}

#[pyfunction]
#[pyo3(signature = (n_samples, test_size=None, train_size=None, shuffle=true, seed=None))]
fn get_split_indices(
    n_samples: usize,
    test_size: Option<f64>,
    train_size: Option<f64>,
    shuffle: bool,
    seed: Option<u64>,
) -> (Vec<usize>, Vec<usize>) {
    core_get_split_indices(n_samples, test_size, train_size, shuffle, seed)
}

#[pymodule]
fn _core(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyStandardScaler>()?;
    m.add_class::<PyMinMaxScaler>()?;
    m.add_class::<PyLabelEncoder>()?;
    m.add_class::<PyOneHotEncoder>()?;
    m.add_function(wrap_pyfunction!(get_split_indices, m)?)?;
    Ok(())
}
```

---

### 4.4. Python Wrappers Package (`thermite/`)

#### `thermite/__init__.py`
```python
"""
Thermite: A Rust-accelerated machine learning library for Python,
drop-in compatible with scikit-learn.
"""

__version__ = "0.1.0"

from . import preprocessing
from . import model_selection

__all__ = [
    "preprocessing",
    "model_selection",
]
```

#### `thermite/preprocessing.py`
```python
import numpy as np
from thermite import _core

class StandardScaler:
    """
    Standardize features by removing the mean and scaling to unit variance.
    """
    def __init__(self, copy=True, with_mean=True, with_std=True):
        self.copy = copy
        self.with_mean = with_mean
        self.with_std = with_std
        self._rust_scaler = None
        self.mean_ = None
        self.var_ = None
        self.scale_ = None
        self.n_samples_seen_ = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array instead.")
        
        self._rust_scaler = _core.StandardScalerRust(self.with_mean, self.with_std)
        self._rust_scaler.fit(X)
        
        self.mean_ = self._rust_scaler.mean()
        self.var_ = self._rust_scaler.var()
        if self.var_ is not None:
            self.scale_ = np.sqrt(self.var_)
        self.n_samples_seen_ = self._rust_scaler.n_samples_seen()
        return self

    def transform(self, X, copy=None):
        if self._rust_scaler is None:
            raise ValueError("This StandardScaler instance is not fitted yet. Call 'fit' before using this estimator.")
        
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array instead.")
        
        should_copy = copy if copy is not None else self.copy
        if should_copy:
            X = X.copy()
            
        return self._rust_scaler.transform(X)

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)


class MinMaxScaler:
    """
    Transform features by scaling each feature to a given range (e.g. [0, 1]).
    """
    def __init__(self, feature_range=(0, 1), copy=True):
        self.feature_range = feature_range
        self.copy = copy
        self._rust_scaler = None
        self.data_min_ = None
        self.data_max_ = None
        self.data_range_ = None
        self.scale_ = None
        self.min_ = None
        self.n_samples_seen_ = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array instead.")
        
        self._rust_scaler = _core.MinMaxScalerRust(self.feature_range[0], self.feature_range[1])
        self._rust_scaler.fit(X)
        
        self.data_min_ = self._rust_scaler.data_min()
        self.data_max_ = self._rust_scaler.data_max()
        self.data_range_ = self._rust_scaler.data_range()
        self.scale_ = self._rust_scaler.scale()
        self.min_ = self._rust_scaler.min()
        self.n_samples_seen_ = self._rust_scaler.n_samples_seen()
        return self

    def transform(self, X):
        if self._rust_scaler is None:
            raise ValueError("This MinMaxScaler instance is not fitted yet. Call 'fit' before using this estimator.")
        
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array instead.")
            
        if self.copy:
            X = X.copy()
            
        return self._rust_scaler.transform(X)

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)


class LabelEncoder:
    """
    Encode target labels with value between 0 and n_classes-1.
    """
    def __init__(self):
        self._rust_encoder = None
        self.classes_ = None

    def fit(self, y):
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError(f"Expected 1D array, got {y.ndim}D array instead.")
            
        self._rust_encoder = _core.LabelEncoderRust()
        self._rust_encoder.fit(y.astype(str).tolist())
        self.classes_ = np.array(self._rust_encoder.classes(), dtype=y.dtype)
        return self

    def transform(self, y):
        if self._rust_encoder is None:
            raise ValueError("This LabelEncoder instance is not fitted yet. Call 'fit' before using this estimator.")
            
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError(f"Expected 1D array, got {y.ndim}D array instead.")
            
        return np.array(self._rust_encoder.transform(y.astype(str).tolist()), dtype=np.intp)

    def fit_transform(self, y):
        return self.fit(y).transform(y)

    def inverse_transform(self, y):
        if self._rust_encoder is None:
            raise ValueError("This LabelEncoder instance is not fitted yet. Call 'fit' before using this estimator.")
            
        y = np.asarray(y, dtype=np.intp)
        if y.ndim != 1:
            raise ValueError(f"Expected 1D array, got {y.ndim}D array instead.")
            
        decoded = self._rust_encoder.inverse_transform(y.tolist())
        return np.array(decoded, dtype=self.classes_.dtype)


class OneHotEncoder:
    """
    Encode categorical features as a one-hot numeric array.
    """
    def __init__(self, categories="auto", drop=None, sparse_output=False, dtype=np.float64):
        self.categories = categories
        self.drop = drop
        self.sparse_output = sparse_output
        self.dtype = dtype
        self._rust_encoder = None
        self.categories_ = None

    def fit(self, X, y=None):
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array instead.")
            
        self._rust_encoder = _core.OneHotEncoderRust(str(self.categories), self.drop, self.sparse_output)
        self._rust_encoder.fit(X.astype(str))
        
        self.categories_ = [np.array(cats, dtype=X.dtype) for cats in self._rust_encoder.categories()]
        return self

    def transform(self, X):
        if self._rust_encoder is None:
            raise ValueError("This OneHotEncoder instance is not fitted yet. Call 'fit' before using this estimator.")
            
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array instead.")
            
        res = self._rust_encoder.transform(X.astype(str))
        return res.astype(self.dtype)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
```

#### `thermite/model_selection.py`
```python
import numpy as np
from thermite import _core

def train_test_split(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None):
    """
    Split arrays or matrices into random train and test subsets.
    """
    if len(arrays) == 0:
        raise ValueError("At least one array required as input")
        
    if test_size is None and train_size is None:
        test_size = 0.25
        
    if stratify is not None:
        raise NotImplementedError("Stratified split is not implemented yet in Thermite.")

    numpy_arrays = [np.asarray(arr) for arr in arrays]
    
    n_samples = len(numpy_arrays[0])
    for arr in numpy_arrays:
        if len(arr) != n_samples:
            raise ValueError("Found input variables with inconsistent numbers of samples")

    seed = random_state if isinstance(random_state, int) else None
    
    train_indices, test_indices = _core.get_split_indices(
        n_samples, test_size, train_size, shuffle, seed
    )
    
    res = []
    for arr in numpy_arrays:
        res.append(arr[train_indices])
        res.append(arr[test_indices])
        
    return res
```

## 5. Verification Method

To independently verify the configuration and packaging, the implementer should perform the following steps:

1. **Verify Cargo Workspace Syntax**:
   Run `cargo check` at the project root directory. This will verify that the root `Cargo.toml` and both child crates (`thermite-core`, `thermite-binding`) parse correctly.

2. **Verify Python Integration and Installation**:
   Initialize and activate a virtual environment, then install Thermite locally via pip or maturin:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install maturin
   maturin develop
   ```
   *Expected Result*: Maturin compiles `crates/thermite-binding/Cargo.toml`, generates the shared library, and links it in the `thermite/` directory as a binary extension module.

3. **Verify Imports & Basic Usage**:
   Run the following Python script to check versioning and mock implementation imports:
   ```python
   import thermite
   from thermite.preprocessing import StandardScaler
   from thermite.model_selection import train_test_split

   print("Thermite version:", thermite.__version__)
   ```
   *Expected Result*: Zero exceptions raised, indicating python package structure and maturin dynamic linking are functional.

4. **Verify Rust Logic**:
   Inside `crates/thermite-core/`, run:
   ```bash
   cargo test
   ```
   *Expected Result*: Local tests execute successfully.
