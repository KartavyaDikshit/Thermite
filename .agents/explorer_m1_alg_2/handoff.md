# Handoff Report — Algorithms & Utilities Design (M1-2 & M1-3)

## 1. Observation
We observed and confirmed the following facts about the codebase and requirements:
* **Cargo Workspace Configuration**: `/Users/kartavyadikshit/Projects/Thermite/Cargo.toml` defines workspace dependencies with exact versions:
  * `ndarray` = "0.15" (with `rayon` feature enabled)
  * `rayon` = "1.8"
  * `rand` = { version = "0.8", features = ["small_rng"] }
  * `pyo3` = { version = "0.21", features = ["extension-module", "abi3-py38"] }
  * `numpy` = "0.21"
* **Module Structure**: 
  * `crates/thermite-core` implements core logic in `lib.rs`, `preprocessing.rs`, and `model_selection.rs` without depending on PyO3.
  * `crates/thermite-binding` compiles PyO3 bindings into the private binary module `_core`.
  * `thermite/` contains Python wrappers delegating to `_core`.
* **Required APIs**:
  1. `train_test_split`: Shuffling & split using `rand` (SmallRng) and `ndarray`, supporting stratified split.
  2. `StandardScaler`: Stores `mean`, `var`, `scale`, and `n_samples_seen`. Parallel variance/mean computation via Rayon.
  3. `MinMaxScaler`: Stores `data_min`, `data_max`, `scale`, and `min`. Parallel bounds computation.
  4. `LabelEncoder`: Support for 1D arrays of integers or strings.
  5. `OneHotEncoder`: Support for 2D categorical features of integers or strings.

---

## 2. Logic Chain

### 2.1 Architectural Rules
* **Core Separation**: Keep `crates/thermite-core` completely free of PyO3 dependencies to make unit tests simple, cargo-compatible, and build-independent of Python runtime.
* **PyO3 Adaptor Layer**: Implement PyO3 class/function translation in `crates/thermite-binding`, converting NumPy arrays (`PyReadonlyArray`) into ndarray views (`ArrayView`) for `thermite-core`.
* **Python Wrappers**: Define the clean user-facing API in `thermite/` (e.g. `thermite.preprocessing.StandardScaler`), validating input shapes and resolving default arguments before invoking the compiled Rust backend.

### 2.2 Component Design
* **`train_test_split`**:
  * Python layer validates matching lengths, parses `test_size`/`train_size`, and extracts seed from `random_state`.
  * Rust layer calculates indices using class-grouped shuffling for stratification or standard shuffling with `SmallRng` for reproducibility.
  * Rust exposes both `split_indices` and a generic `split_arrays` using Python's highly optimized NumPy `.take(indices, axis=0)` method via PyO3 to avoid dtype monomorphization overhead in Rust.
* **`StandardScaler` & `MinMaxScaler`**:
  * Parallelized column-wise calculations using Rayon's `into_par_iter()` over column indices.
  * Row-wise transformation parallelized using Rayon over the rows of the `ndarray::Array2`.
  * Boundary conditions (e.g., zero variance or constant features) handled safely by mapping scale to 1.0 (StandardScaler) or scale to 0.0 & min to `feature_range[0]` (MinMaxScaler), matching scikit-learn behavior.
* **`LabelEncoder` & `OneHotEncoder`**:
  * Support for string/integer conversions by treating values as strings during lookups inside Rust, mapping unique sorted categories to indices using `HashMap` structures.
  * Python wrapper preserves type compatibility of public attributes (e.g., `classes_` and `categories_`) by casting string values back to original dtypes.
  * Rayon used to execute categorical row lookups concurrently during `transform`.

---

## 3. Caveats
* **Stratification Adjustment**: If stratified group rounding doesn't exactly match `n_test` due to integer division, the remainder of indices is trimmed/padded to guarantee the exact length contract requested by the user, prioritizing class-balance.
* **Object Array Conversion**: One-hot and label encoding for custom Python objects or mixed types is normalized to strings to ensure high-performance Rust processing.
* **Maturin cdylib unit testing**: Unit tests for binding logic should be tested via PyTest under `tests/` since Cargo cannot link PyO3 symbols without a Python runner. Pure math core logic is fully unit-testable via `cargo test`.

---

## 4. Conclusion & Templates

The designed templates provide a complete blueprint for implementing M1-2 and M1-3.

### 4.1 `crates/thermite-core/src/model_selection.rs`
```rust
use rand::prelude::*;
use rand::rngs::SmallRng;
use std::collections::HashMap;

/// Shuffles and splits indices for train_test_split.
/// Handles optional stratified split by grouping indices by label.
pub fn split_indices(
    n_samples: usize,
    n_train: usize,
    n_test: usize,
    seed: Option<u64>,
    shuffle: bool,
    stratify: Option<Vec<String>>,
) -> (Vec<usize>, Vec<usize>) {
    let mut rng = if let Some(s) = seed {
        SmallRng::seed_from_u64(s)
    } else {
        SmallRng::from_entropy()
    };

    if let Some(stratify_labels) = stratify {
        // Stratified split: group indices by class label
        let mut class_groups: HashMap<String, Vec<usize>> = HashMap::new();
        for (idx, label) in stratify_labels.iter().enumerate() {
            class_groups.entry(label.clone()).or_default().push(idx);
        }

        let mut train_indices = Vec::new();
        let mut test_indices = Vec::new();

        let ratio = n_test as f64 / n_samples as f64;

        for (_, mut group) in class_groups {
            if shuffle {
                group.shuffle(&mut rng);
            }

            let group_test_size = ((group.len() as f64 * ratio).round() as usize)
                .min(group.len())
                .max(if group.len() > 0 && n_test > 0 { 1 } else { 0 });

            test_indices.extend_from_slice(&group[..group_test_size]);
            train_indices.extend_from_slice(&group[group_test_size..]);
        }

        if shuffle {
            train_indices.shuffle(&mut rng);
            test_indices.shuffle(&mut rng);
        }

        // Adjust sizes if rounding caused deviations from the exact counts requested
        if train_indices.len() + test_indices.len() == n_samples {
            while test_indices.len() > n_test && !test_indices.is_empty() {
                train_indices.push(test_indices.pop().unwrap());
            }
            while test_indices.len() < n_test && !train_indices.is_empty() {
                test_indices.push(train_indices.pop().unwrap());
            }
        }

        (train_indices, test_indices)
    } else {
        // Standard shuffle/split
        let mut indices: Vec<usize> = (0..n_samples).collect();
        if shuffle {
            indices.shuffle(&mut rng);
        }
        let train_indices = indices[..n_train].to_vec();
        let test_indices = indices[n_train..(n_train + n_test)].to_vec();
        (train_indices, test_indices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_indices_simple() {
        let (train, test) = split_indices(10, 8, 2, Some(42), true, None);
        assert_eq!(train.len(), 8);
        assert_eq!(test.len(), 2);
        for t in &test {
            assert!(!train.contains(t));
        }
    }

    #[test]
    fn test_split_indices_stratify() {
        let stratify = vec![
            "A".to_string(), "A".to_string(), "A".to_string(), "A".to_string(),
            "B".to_string(), "B".to_string(), "B".to_string(), "B".to_string(),
            "B".to_string(), "B".to_string()
        ];
        let (train, test) = split_indices(10, 5, 5, Some(42), true, Some(stratify));
        assert_eq!(train.len(), 5);
        assert_eq!(test.len(), 5);
    }
}
```

### 4.2 `crates/thermite-core/src/preprocessing.rs`
```rust
use ndarray::prelude::*;
use rayon::prelude::*;

/// Fit StandardScaler by calculating mean and variance of each column in parallel
pub fn fit_standard_scaler(
    X: &ArrayView2<f64>,
    with_mean: bool,
    with_std: bool,
) -> (Option<Array1<f64>>, Option<Array1<f64>>, Option<Array1<f64>>, usize) {
    let n_samples = X.nrows();
    let n_features = X.ncols();

    if n_samples == 0 {
        return (None, None, None, 0);
    }

    let stats: Vec<(f64, f64)> = (0..n_features)
        .into_par_iter()
        .map(|col_idx| {
            let col = X.column(col_idx);
            let sum: f64 = col.sum();
            let mean = sum / n_samples as f64;

            let var_sum: f64 = col.iter().map(|&x| {
                let diff = x - mean;
                diff * diff
            }).sum();
            let var = var_sum / n_samples as f64;

            (mean, var)
        })
        .collect();

    let mean = if with_mean {
        Some(Array1::from_shape_fn(n_features, |i| stats[i].0))
    } else {
        None
    };

    let var = if with_std {
        Some(Array1::from_shape_fn(n_features, |i| stats[i].1))
    } else {
        None
    };

    let scale = if with_std {
        let scale_arr = Array1::from_shape_fn(n_features, |i| {
            let v = stats[i].1;
            if v <= 1e-18 {
                1.0
            } else {
                v.sqrt()
            }
        });
        Some(scale_arr)
    } else {
        None
    };

    (mean, var, scale, n_samples)
}

/// Apply StandardScaler transformation in parallel row-by-row
pub fn transform_standard_scaler(
    X: &ArrayView2<f64>,
    mean: Option<ArrayView1<f64>>,
    scale: Option<ArrayView1<f64>>,
) -> Array2<f64> {
    let mut Y = X.to_owned();

    Y.axis_iter_mut(Axis(0))
        .into_par_iter()
        .for_each(|mut row| {
            if let Some(m) = mean {
                row -= &m;
            }
            if let Some(s) = scale {
                row /= &s;
            }
        });

    Y
}

/// Apply inverse StandardScaler transformation in parallel row-by-row
pub fn inverse_transform_standard_scaler(
    X: &ArrayView2<f64>,
    mean: Option<ArrayView1<f64>>,
    scale: Option<ArrayView1<f64>>,
) -> Array2<f64> {
    let mut Y = X.to_owned();

    Y.axis_iter_mut(Axis(0))
        .into_par_iter()
        .for_each(|mut row| {
            if let Some(s) = scale {
                row *= &s;
            }
            if let Some(m) = mean {
                row += &m;
            }
        });

    Y
}

/// Fit MinMaxScaler by finding min and max for each column in parallel
pub fn fit_min_max_scaler(
    X: &ArrayView2<f64>,
    feature_range: (f64, f64),
) -> (Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>) {
    let n_samples = X.nrows();
    let n_features = X.ncols();

    let bounds: Vec<(f64, f64)> = (0..n_features)
        .into_par_iter()
        .map(|col_idx| {
            let col = X.column(col_idx);
            let mut min_val = f64::INFINITY;
            let mut max_val = f64::NEG_INFINITY;
            for &x in col.iter() {
                if x < min_val { min_val = x; }
                if x > max_val { max_val = x; }
            }
            (min_val, max_val)
        })
        .collect();

    let data_min = Array1::from_shape_fn(n_features, |i| bounds[i].0);
    let data_max = Array1::from_shape_fn(n_features, |i| bounds[i].1);

    let (fr_min, fr_max) = feature_range;
    let fr_range = fr_max - fr_min;

    let mut scale = Array1::zeros(n_features);
    let mut min = Array1::zeros(n_features);

    for i in 0..n_features {
        let diff = data_max[i] - data_min[i];
        if diff <= 1e-18 {
            scale[i] = 0.0;
            min[i] = fr_min;
        } else {
            scale[i] = fr_range / diff;
            min[i] = fr_min - data_min[i] * scale[i];
        }
    }

    (data_min, data_max, scale, min)
}

pub fn transform_min_max_scaler(
    X: &ArrayView2<f64>,
    scale: &ArrayView1<f64>,
    min: &ArrayView1<f64>,
) -> Array2<f64> {
    let mut Y = X.to_owned();
    Y.axis_iter_mut(Axis(0))
        .into_par_iter()
        .for_each(|mut row| {
            row *= scale;
            row += min;
        });
    Y
}

pub fn inverse_transform_min_max_scaler(
    X: &ArrayView2<f64>,
    scale: &ArrayView1<f64>,
    min: &ArrayView1<f64>,
    data_min: &ArrayView1<f64>,
) -> Array2<f64> {
    let mut Y = X.to_owned();
    let n_features = Y.ncols();

    Y.axis_iter_mut(Axis(0))
        .into_par_iter()
        .for_each(|mut row| {
            for j in 0..n_features {
                if scale[j] == 0.0 {
                    row[j] = data_min[j];
                } else {
                    row[j] = (row[j] - min[j]) / scale[j];
                }
            }
        });
    Y
}
```

### 4.3 `crates/thermite-binding/src/lib.rs`
```rust
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use numpy::{PyArray1, PyArray2, PyReadonlyArray2, ToPyArray};
use std::collections::HashMap;
use rayon::prelude::*;

#[pyfunction]
fn ping() -> PyResult<String> {
    Ok("pong".to_string())
}

#[pyfunction]
fn split_indices(
    n_samples: usize,
    n_train: usize,
    n_test: usize,
    seed: Option<u64>,
    shuffle: bool,
    stratify: Option<Vec<String>>,
) -> PyResult<(Vec<usize>, Vec<usize>)> {
    let (train, test) = thermite_core::model_selection::split_indices(
        n_samples, n_train, n_test, seed, shuffle, stratify
    );
    Ok((train, test))
}

#[pyfunction]
fn split_arrays<'py>(
    py: Python<'py>,
    arrays: Vec<Bound<'py, PyAny>>,
    train_indices: Vec<usize>,
    test_indices: Vec<usize>,
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let mut result = Vec::with_capacity(arrays.len() * 2);
    let train_py = PyArray1::from_iter(py, train_indices);
    let test_py = PyArray1::from_iter(py, test_indices);

    for arr in arrays {
        let train_split = arr.call_method1("take", (train_py.clone(), 0))?;
        let test_split = arr.call_method1("take", (test_py.clone(), 0))?;
        result.push(train_split);
        result.push(test_split);
    }

    Ok(result)
}

#[pyclass]
#[derive(Clone)]
pub struct StandardScaler {
    #[pyo3(get)]
    pub mean: Option<Py<PyArray1<f64>>>,
    #[pyo3(get)]
    pub var: Option<Py<PyArray1<f64>>>,
    #[pyo3(get)]
    pub scale: Option<Py<PyArray1<f64>>>,
    #[pyo3(get)]
    pub n_samples_seen: usize,
    pub with_mean: bool,
    pub with_std: bool,
}

#[pymethods]
impl StandardScaler {
    #[new]
    #[pyo3(signature = (with_mean=true, with_std=true))]
    fn new(with_mean: bool, with_std: bool) -> Self {
        Self {
            mean: None,
            var: None,
            scale: None,
            n_samples_seen: 0,
            with_mean,
            with_std,
        }
    }

    fn fit(&mut self, X: PyReadonlyArray2<f64>) -> PyResult<()> {
        let x_view = X.as_array();
        let (mean, var, scale, n_samples) = thermite_core::preprocessing::fit_standard_scaler(
            &x_view,
            self.with_mean,
            self.with_std,
        );

        Python::with_gil(|py| {
            self.mean = mean.map(|m| m.to_pyarray(py).unbind());
            self.var = var.map(|v| v.to_pyarray(py).unbind());
            self.scale = scale.map(|s| s.to_pyarray(py).unbind());
        });
        self.n_samples_seen = n_samples;

        Ok(())
    }

    fn transform<'py>(&self, py: Python<'py>, X: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x_view = X.as_array();
        let mean_arr = self.mean.as_ref().map(|m| m.bind(py).as_array().to_owned());
        let scale_arr = self.scale.as_ref().map(|s| s.bind(py).as_array().to_owned());

        let transformed = thermite_core::preprocessing::transform_standard_scaler(
            &x_view,
            mean_arr.as_ref().map(|m| m.view()),
            scale_arr.as_ref().map(|s| s.view()),
        );

        Ok(transformed.to_pyarray(py))
    }

    fn inverse_transform<'py>(&self, py: Python<'py>, X: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x_view = X.as_array();
        let mean_arr = self.mean.as_ref().map(|m| m.bind(py).as_array().to_owned());
        let scale_arr = self.scale.as_ref().map(|s| s.bind(py).as_array().to_owned());

        let inverted = thermite_core::preprocessing::inverse_transform_standard_scaler(
            &x_view,
            mean_arr.as_ref().map(|m| m.view()),
            scale_arr.as_ref().map(|s| s.view()),
        );

        Ok(inverted.to_pyarray(py))
    }
}

#[pyclass]
#[derive(Clone)]
pub struct MinMaxScaler {
    #[pyo3(get)]
    pub data_min: Option<Py<PyArray1<f64>>>,
    #[pyo3(get)]
    pub data_max: Option<Py<PyArray1<f64>>>,
    #[pyo3(get)]
    pub scale: Option<Py<PyArray1<f64>>>,
    #[pyo3(get)]
    pub min: Option<Py<PyArray1<f64>>>,
    pub feature_range: (f64, f64),
}

#[pymethods]
impl MinMaxScaler {
    #[new]
    #[pyo3(signature = (feature_range=(0.0, 1.0)))]
    fn new(feature_range: (f64, f64)) -> Self {
        Self {
            data_min: None,
            data_max: None,
            scale: None,
            min: None,
            feature_range,
        }
    }

    fn fit(&mut self, X: PyReadonlyArray2<f64>) -> PyResult<()> {
        let x_view = X.as_array();
        let (data_min, data_max, scale, min) = thermite_core::preprocessing::fit_min_max_scaler(
            &x_view,
            self.feature_range,
        );

        Python::with_gil(|py| {
            self.data_min = Some(data_min.to_pyarray(py).unbind());
            self.data_max = Some(data_max.to_pyarray(py).unbind());
            self.scale = Some(scale.to_pyarray(py).unbind());
            self.min = Some(min.to_pyarray(py).unbind());
        });

        Ok(())
    }

    fn transform<'py>(&self, py: Python<'py>, X: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x_view = X.as_array();
        let scale_arr = self.scale.as_ref().ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Not fitted"))?
            .bind(py).as_array().to_owned();
        let min_arr = self.min.as_ref().unwrap().bind(py).as_array().to_owned();

        let transformed = thermite_core::preprocessing::transform_min_max_scaler(
            &x_view,
            &scale_arr.view(),
            &min_arr.view(),
        );

        Ok(transformed.to_pyarray(py))
    }

    fn inverse_transform<'py>(&self, py: Python<'py>, X: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x_view = X.as_array();
        let scale_arr = self.scale.as_ref().ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Not fitted"))?
            .bind(py).as_array().to_owned();
        let min_arr = self.min.as_ref().unwrap().bind(py).as_array().to_owned();
        let data_min_arr = self.data_min.as_ref().unwrap().bind(py).as_array().to_owned();

        let inverted = thermite_core::preprocessing::inverse_transform_min_max_scaler(
            &x_view,
            &scale_arr.view(),
            &min_arr.view(),
            &data_min_arr.view(),
        );

        Ok(inverted.to_pyarray(py))
    }
}

#[pyclass]
pub struct LabelEncoder {
    #[pyo3(get)]
    pub classes: Option<PyObject>,
    pub int_map: Option<HashMap<i64, usize>>,
    pub str_map: Option<HashMap<String, usize>>,
    pub classes_int: Option<Vec<i64>>,
    pub classes_str: Option<Vec<String>>,
}

#[pymethods]
impl LabelEncoder {
    #[new]
    fn new() -> Self {
        Self {
            classes: None,
            int_map: None,
            str_map: None,
            classes_int: None,
            classes_str: None,
        }
    }

    fn fit_int(&mut self, y: Vec<i64>) -> PyResult<()> {
        let mut unique_vals = y;
        unique_vals.sort_unstable();
        unique_vals.dedup();

        let mut int_map = HashMap::with_capacity(unique_vals.len());
        for (idx, &val) in unique_vals.iter().enumerate() {
            int_map.insert(val, idx);
        }

        self.classes_int = Some(unique_vals.clone());
        self.int_map = Some(int_map);
        Python::with_gil(|py| {
            self.classes = Some(unique_vals.to_pyarray(py).to_owned().into());
        });

        Ok(())
    }

    fn fit_str(&mut self, y: Vec<String>) -> PyResult<()> {
        let mut unique_vals = y;
        unique_vals.sort_unstable();
        unique_vals.dedup();

        let mut str_map = HashMap::with_capacity(unique_vals.len());
        for (idx, val) in unique_vals.iter().enumerate() {
            str_map.insert(val.clone(), idx);
        }

        self.classes_str = Some(unique_vals.clone());
        self.str_map = Some(str_map);
        Python::with_gil(|py| {
            self.classes = Some(unique_vals.to_object(py));
        });

        Ok(())
    }

    fn transform_int(&self, y: Vec<i64>) -> PyResult<Vec<i64>> {
        let int_map = self.int_map.as_ref().ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Not fitted"))?;
        let res: Result<Vec<i64>, i64> = y.par_iter()
            .map(|&val| int_map.get(&val).map(|&idx| idx as i64).ok_or(val))
            .collect();

        match res {
            Ok(indices) => Ok(indices),
            Err(unseen) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("y contains previously unseen labels: {}", unseen)
            ))
        }
    }

    fn transform_str(&self, y: Vec<String>) -> PyResult<Vec<i64>> {
        let str_map = self.str_map.as_ref().ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Not fitted"))?;
        let res: Result<Vec<i64>, String> = y.par_iter()
            .map(|val| str_map.get(val).map(|&idx| idx as i64).ok_or_else(|| val.clone()))
            .collect();

        match res {
            Ok(indices) => Ok(indices),
            Err(unseen) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("y contains previously unseen labels: '{}'", unseen)
            ))
        }
    }

    fn inverse_transform<'py>(&self, py: Python<'py>, y: Vec<i64>) -> PyResult<Bound<'py, PyAny>> {
        if let Some(classes_int) = &self.classes_int {
            let res: Result<Vec<i64>, i64> = y.par_iter()
                .map(|&idx| {
                    if idx >= 0 && (idx as usize) < classes_int.len() {
                        Ok(classes_int[idx as usize])
                    } else {
                        Err(idx)
                    }
                })
                .collect();
            match res {
                Ok(vals) => Ok(vals.to_pyarray(py).into_any()),
                Err(invalid_idx) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("y contains invalid index: {}", invalid_idx)
                ))
            }
        } else if let Some(classes_str) = &self.classes_str {
            let res: Result<Vec<String>, i64> = y.par_iter()
                .map(|&idx| {
                    if idx >= 0 && (idx as usize) < classes_str.len() {
                        Ok(classes_str[idx as usize].clone())
                    } else {
                        Err(idx)
                    }
                })
                .collect();
            match res {
                Ok(vals) => Ok(vals.to_object(py).into_bound(py)),
                Err(invalid_idx) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("y contains invalid index: {}", invalid_idx)
                ))
            }
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Not fitted"))
        }
    }
}

#[pyclass]
pub struct OneHotEncoder {
    #[pyo3(get)]
    pub categories: Option<Vec<Vec<String>>>,
    pub maps: Option<Vec<HashMap<String, usize>>>,
    pub offsets: Option<Vec<usize>>,
    pub total_features: usize,
    pub handle_unknown: String,
}

#[pymethods]
impl OneHotEncoder {
    #[new]
    #[pyo3(signature = (handle_unknown="error".to_string()))]
    fn new(handle_unknown: String) -> Self {
        Self {
            categories: None,
            maps: None,
            offsets: None,
            total_features: 0,
            handle_unknown,
        }
    }

    fn fit(&mut self, X: Vec<Vec<String>>) -> PyResult<()> {
        let n_features = X.len();
        if n_features == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Empty input"));
        }

        let mut categories = Vec::with_capacity(n_features);
        let mut maps = Vec::with_capacity(n_features);
        let mut offsets = Vec::with_capacity(n_features);
        let mut current_offset = 0;

        for col in X {
            let mut unique_cats = col;
            unique_cats.sort_unstable();
            unique_cats.dedup();

            let mut map = HashMap::with_capacity(unique_cats.len());
            for (idx, cat) in unique_cats.iter().enumerate() {
                map.insert(cat.clone(), idx);
            }

            offsets.push(current_offset);
            current_offset += unique_cats.len();

            categories.push(unique_cats);
            maps.push(map);
        }

        self.categories = Some(categories);
        self.maps = Some(maps);
        self.offsets = Some(offsets);
        self.total_features = current_offset;

        Ok(())
    }

    fn transform<'py>(&self, py: Python<'py>, X: Vec<Vec<String>>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let categories = self.categories.as_ref().ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Not fitted"))?;
        let maps = self.maps.as_ref().unwrap();
        let offsets = self.offsets.as_ref().unwrap();

        let n_features = X.len();
        if n_features != categories.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Signature mismatch: expected {} features, got {}", categories.len(), n_features)
            ));
        }

        let n_samples = X[0].len();
        for col in &X {
            if col.len() != n_samples {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Inconsistent column lengths"));
            }
        }

        let mut out = ndarray::Array2::<f64>::zeros((n_samples, self.total_features));
        let mut rows: Vec<_> = out.axis_iter_mut(ndarray::Axis(0)).collect();
        let handle_unknown = &self.handle_unknown;

        let res: Result<(), (usize, usize, String)> = rows.par_iter_mut()
            .enumerate()
            .map(|(i, mut row)| {
                for j in 0..n_features {
                    let val = &X[j][i];
                    if let Some(&idx) = maps[j].get(val) {
                        row[offsets[j] + idx] = 1.0;
                    } else if handle_unknown == "error" {
                        return Err((i, j, val.clone()));
                    }
                }
                Ok(())
            })
            .collect();

        match res {
            Ok(_) => Ok(out.to_pyarray(py)),
            Err((sample_idx, feature_idx, val)) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Found unknown category '{}' in column {} during transform at sample {}", val, feature_idx, sample_idx)
            ))
        }
    }

    fn inverse_transform<'py>(&self, X: PyReadonlyArray2<f64>) -> PyResult<Vec<Vec<String>>> {
        let categories = self.categories.as_ref().ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Not fitted"))?;
        let offsets = self.offsets.as_ref().unwrap();

        let x_view = X.as_array();
        let n_samples = x_view.nrows();
        let n_features = categories.len();

        if x_view.ncols() != self.total_features {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Input shape mismatch: expected {} columns, got {}", self.total_features, x_view.ncols())
            ));
        }

        let mut result = vec![vec![String::new(); n_samples]; n_features];

        for j in 0..n_features {
            let start = offsets[j];
            let end = if j + 1 < n_features { offsets[j + 1] } else { self.total_features };

            for i in 0..n_samples {
                let mut max_val = x_view[[i, start]];
                let mut max_idx = 0;
                for idx in 1..(end - start) {
                    let val = x_view[[i, start + idx]];
                    if val > max_val {
                        max_val = val;
                        max_idx = idx;
                    }
                }
                result[j][i] = categories[j][max_idx].clone();
            }
        }

        Ok(result)
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ping, m)?)?;
    m.add_function(wrap_pyfunction!(split_indices, m)?)?;
    m.add_function(wrap_pyfunction!(split_arrays, m)?)?;
    m.add_class::<StandardScaler>()?;
    m.add_class::<MinMaxScaler>()?;
    m.add_class::<LabelEncoder>()?;
    m.add_class::<OneHotEncoder>()?;
    Ok(())
}
```

### 4.4 `thermite/model_selection.py`
```python
import numpy as np
from thermite import _core

def train_test_split(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None):
    if not arrays:
        raise ValueError("At least one array required as input")

    np_arrays = [np.asarray(arr) for arr in arrays]
    n_samples = len(np_arrays[0])

    for arr in np_arrays:
        if len(arr) != n_samples:
            raise ValueError("Found input variables with inconsistent numbers of samples")

    if test_size is None and train_size is None:
        test_size = 0.25

    n_test = None
    n_train = None

    if test_size is not None:
        if isinstance(test_size, float):
            if not (0.0 < test_size < 1.0):
                raise ValueError(f"test_size={test_size} should be between 0.0 and 1.0")
            n_test = int(np.ceil(test_size * n_samples))
        elif isinstance(test_size, int):
            if not (0 <= test_size <= n_samples):
                raise ValueError(f"test_size={test_size} should be between 0 and {n_samples}")
            n_test = test_size
        else:
            raise TypeError("test_size must be int or float")

    if train_size is not None:
        if isinstance(train_size, float):
            if not (0.0 < train_size < 1.0):
                raise ValueError(f"train_size={train_size} should be between 0.0 and 1.0")
            n_train = int(np.floor(train_size * n_samples))
        elif isinstance(train_size, int):
            if not (0 <= train_size <= n_samples):
                raise ValueError(f"train_size={train_size} should be between 0 and {n_samples}")
            n_train = train_size
        else:
            raise TypeError("train_size must be int or float")

    if n_test is None:
        n_test = n_samples - n_train
    elif n_train is None:
        n_train = n_samples - n_test

    if n_train + n_test > n_samples:
        raise ValueError(
            f"The sum of train_size and test_size ({n_train + n_test}) "
            f"should be smaller than the number of samples ({n_samples})"
        )

    seed = None
    if random_state is not None:
        if isinstance(random_state, int):
            seed = random_state & 0xFFFFFFFF
        elif hasattr(random_state, "randint"):
            seed = int(random_state.randint(0, 2**32 - 1))
        else:
            raise ValueError(f"{random_state} cannot be used to seed a RandomState instance")

    stratify_list = None
    if stratify is not None:
        stratify_arr = np.asarray(stratify)
        if len(stratify_arr) != n_samples:
            raise ValueError("stratify must have the same length as the input arrays")
        stratify_list = [str(x) for x in stratify_arr]

    # Generate indices
    train_idx, test_idx = _core.split_indices(
        n_samples, n_train, n_test, seed, shuffle, stratify_list
    )

    # Split arrays
    return _core.split_arrays(np_arrays, train_idx, test_idx)
```

### 4.5 `thermite/preprocessing.py`
```python
import numpy as np
from thermite import _core

class StandardScaler:
    def __init__(self, *, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std
        self._scaler = _core.StandardScaler(with_mean, with_std)

    @property
    def mean_(self):
        m = self._scaler.mean
        if m is None:
            raise AttributeError("This StandardScaler instance is not fitted yet.")
        return m

    @property
    def var_(self):
        v = self._scaler.var
        if v is None:
            raise AttributeError("This StandardScaler instance is not fitted yet.")
        return v

    @property
    def scale_(self):
        s = self._scaler.scale
        if s is None:
            raise AttributeError("This StandardScaler instance is not fitted yet.")
        return s

    @property
    def n_samples_seen_(self):
        return self._scaler.n_samples_seen

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array instead")
        self._scaler.fit(X)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array instead")
        if self._scaler.n_samples_seen == 0:
            raise AttributeError("This StandardScaler instance is not fitted yet.")
        return self._scaler.transform(X)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array instead")
        if self._scaler.n_samples_seen == 0:
            raise AttributeError("This StandardScaler instance is not fitted yet.")
        return self._scaler.inverse_transform(X)


class MinMaxScaler:
    def __init__(self, feature_range=(0.0, 1.0)):
        if len(feature_range) != 2 or feature_range[0] >= feature_range[1]:
            raise ValueError("feature_range must be a tuple (min, max) where min < max")
        self.feature_range = (float(feature_range[0]), float(feature_range[1]))
        self._scaler = _core.MinMaxScaler(self.feature_range)

    @property
    def data_min_(self):
        val = self._scaler.data_min
        if val is None:
            raise AttributeError("This MinMaxScaler instance is not fitted yet.")
        return val

    @property
    def data_max_(self):
        val = self._scaler.data_max
        if val is None:
            raise AttributeError("This MinMaxScaler instance is not fitted yet.")
        return val

    @property
    def scale_(self):
        val = self._scaler.scale
        if val is None:
            raise AttributeError("This MinMaxScaler instance is not fitted yet.")
        return val

    @property
    def min_(self):
        val = self._scaler.min
        if val is None:
            raise AttributeError("This MinMaxScaler instance is not fitted yet.")
        return val

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array instead")
        self._scaler.fit(X)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array instead")
        if self._scaler.data_min is None:
            raise AttributeError("This MinMaxScaler instance is not fitted yet.")
        return self._scaler.transform(X)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array instead")
        if self._scaler.data_min is None:
            raise AttributeError("This MinMaxScaler instance is not fitted yet.")
        return self._scaler.inverse_transform(X)


class LabelEncoder:
    def __init__(self):
        self._encoder = _core.LabelEncoder()

    @property
    def classes_(self):
        classes = self._encoder.classes
        if classes is None:
            raise AttributeError("This LabelEncoder instance is not fitted yet.")
        return classes

    def fit(self, y):
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError(f"y should be a 1D array, got shape {y.shape}")

        if np.issubdtype(y.dtype, np.integer):
            y_int = y.astype(np.int64).tolist()
            self._encoder.fit_int(y_int)
        else:
            y_str = [str(x) for x in y]
            self._encoder.fit_str(y_str)
        return self

    def transform(self, y):
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError(f"y should be a 1D array, got shape {y.shape}")

        if self._encoder.classes is None:
            raise AttributeError("This LabelEncoder instance is not fitted yet.")

        if np.issubdtype(y.dtype, np.integer):
            y_int = y.astype(np.int64).tolist()
            res = self._encoder.transform_int(y_int)
        else:
            y_str = [str(x) for x in y]
            res = self._encoder.transform_str(y_str)
        return np.array(res, dtype=np.int64)

    def fit_transform(self, y):
        return self.fit(y).transform(y)

    def inverse_transform(self, y):
        y = np.asarray(y, dtype=np.int64)
        if y.ndim != 1:
            raise ValueError(f"y should be a 1D array, got shape {y.shape}")

        if self._encoder.classes is None:
            raise AttributeError("This LabelEncoder instance is not fitted yet.")

        res = self._encoder.inverse_transform(y.tolist())
        return np.asarray(res)


class OneHotEncoder:
    def __init__(self, handle_unknown="error"):
        if handle_unknown not in ("error", "ignore"):
            raise ValueError("handle_unknown must be 'error' or 'ignore'")
        self.handle_unknown = handle_unknown
        self._encoder = _core.OneHotEncoder(handle_unknown)
        self.categories_ = None
        self._orig_dtypes = None

    def fit(self, X, y=None):
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array instead")

        n_samples, n_features = X.shape
        self._orig_dtypes = [X[:, j].dtype for j in range(n_features)]

        X_str = [[str(x) for x in X[:, j]] for j in range(n_features)]
        self._encoder.fit(X_str)

        raw_cats = self._encoder.categories
        self.categories_ = []
        for j, col_cats in enumerate(raw_cats):
            dtype = self._orig_dtypes[j]
            try:
                cat_arr = np.array(col_cats, dtype=dtype)
            except (ValueError, TypeError):
                cat_arr = np.array(col_cats, dtype=object)
            self.categories_.append(cat_arr)

        return self

    def transform(self, X):
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array instead")
        if self.categories_ is None:
            raise AttributeError("This OneHotEncoder instance is not fitted yet.")

        n_samples, n_features = X.shape
        if n_features != len(self.categories_):
            raise ValueError(f"X has {n_features} features, but OneHotEncoder is expecting {len(self.categories_)} features.")

        X_str = [[str(x) for x in X[:, j]] for j in range(n_features)]
        return self._encoder.transform(X_str)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array instead")
        if self.categories_ is None:
            raise AttributeError("This OneHotEncoder instance is not fitted yet.")

        res_str = self._encoder.inverse_transform(X)
        n_features = len(res_str)

        out_columns = []
        for j in range(n_features):
            dtype = self._orig_dtypes[j]
            col_data = res_str[j]
            try:
                arr = np.array(col_data, dtype=dtype)
            except (ValueError, TypeError):
                arr = np.array(col_data, dtype=object)
            out_columns.append(arr)

        return np.column_stack(out_columns)
```

---

## 5. Verification Method

To verify correctness and performance:
1. **Compilation**: Run `maturin develop` inside `.venv` to build the new binding API.
2. **Crate tests**: Run `cargo test` in the root workspace.
   * Verify that `thermite-core` tests pass without errors.
3. **Integration and End-to-End Tests**:
   * Add a new PyTest suite `tests/test_preprocessing_selection.py` containing:
     * Correctness test checking that `StandardScaler` matches `sklearn.preprocessing.StandardScaler` output to a tolerance of `1e-7`.
     * Correctness test checking that `MinMaxScaler` matches `sklearn.preprocessing.MinMaxScaler` output to a tolerance of `1e-7`.
     * Split index distributions for stratified split inside `train_test_split`.
     * Categorical encoding tests for strings and integers using `LabelEncoder` and `OneHotEncoder`.
4. **Command to run**:
   ```bash
   pytest tests/test_preprocessing_selection.py
   ```
