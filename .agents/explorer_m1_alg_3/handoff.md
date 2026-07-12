# Handoff Report: Preprocessing & Model Selection Design (M1-2 & M1-3)

This report details the architectural analysis, design decisions, and recommended code templates for implementing the core algorithms and PyO3 bindings for Milestones M1-2 (`train_test_split`) and M1-3 (`StandardScaler`, `MinMaxScaler`, `LabelEncoder`, `OneHotEncoder`).

---

## 1. Observation

Direct observations from the workspace files and structure:
- **Build Setup & Packages (M1-1)**: Done. Cargo workspace configuration (`Cargo.toml`) is in the root, and private compiled binaries are exposed at `thermite._core` via Maturin (`pyproject.toml`).
- **Target Crate Layout**:
  - `crates/thermite-core/src/lib.rs` (Workspace library exposing core logic)
  - `crates/thermite-core/src/preprocessing.rs` (Placeholder for preprocessing algorithms)
  - `crates/thermite-core/src/model_selection.rs` (Placeholder for split utilities)
  - `crates/thermite-binding/src/lib.rs` (PyO3 binding module exposing private `_core` API)
  - `thermite/preprocessing.py` (Python class wrappers around private Rust bindings)
  - `thermite/model_selection.py` (Python wrappers for split utilities)
- **E2E Test Specifications**:
  - `tests/test_tier1_preprocessing.py` defines the precise API and parameters for StandardScaler (with_mean, with_std, mean_), MinMaxScaler (feature_range, data_min_, data_max_, scale_, min_), LabelEncoder (classes_, numeric/string support, exception raising), and OneHotEncoder (handle_unknown='ignore', drop='first', custom categories, inverse_transform).
  - `tests/test_tier1_model_selection_pipeline.py` defines the API and parameters for train_test_split (stratify, custom size/percentage splitting, random state reproducibility, multiple inputs splitting).

---

## 2. Logic Chain

Based on these observations, the design must satisfy the following:
1. **`train_test_split` Separation of Concerns**:
   - Instead of trying to serialize arbitrary multi-dimensional list types (e.g. `f64`, `i32`, objects, strings) from Python to Rust, we can compute indices of train/test splits inside Rust using `rand::rngs::SmallRng` and return the indices `(Vec<usize>, Vec<usize>)` to Python.
   - The Python wrapper uses NumPy advanced indexing to slice the input arrays. This leverages highly optimized C-level memory copying in Python while maintaining Rust-based deterministic shuffling and stratified partitioning.
2. **Stratification via Hamiltonian Largest Remainder**:
   - In order to keep class proportions exact across train and test splits, we group the indices of the stratified labels by class.
   - We distribute the target test sample count proportional to the class sizes. To resolve rounding fractional counts exactly, we apply the Largest Remainder Method (Hamilton method) to assign leftovers to classes with the largest remainders. This ensures the output test/train arrays match the exact specified dimensions.
3. **Rust `CategoricalValue` for Flexible Encoders**:
   - Since `LabelEncoder` and `OneHotEncoder` must accept both strings and integers, we define a unified custom enum `CategoricalValue { Int(i64), Str(String) }` in Rust.
   - Implementing PyO3's `FromPyObject` and `IntoPyObject` traits for `CategoricalValue` allows PyO3 to seamlessly convert Python integers, strings, or mixed list objects into Rust's vector representations.
4. **Rayon Column-Wise and Sample-Wise Parallelization**:
   - For `StandardScaler` and `MinMaxScaler`, statistics are calculated independently per column (feature). We convert columns into parallel iterors using Rayon's `par_iter()` for parallelized mean, variance, min, and max calculations.
   - For transformations, we apply Rayon's `par_bridge()` or `par_chunks_mut()` to perform in-place column mutations and row transformations in parallel.
5. **Handling Corner Cases**:
   - *Constant Features (variance/range of zero)*: StandardScaler sets scale to 1.0; MinMaxScaler sets scale to 0.0 and maps values to `feature_range[0]`.
   - *Unseen categories in OneHotEncoder*: Under `handle_unknown='ignore'`, unseen categories result in all-zero columns. Under `handle_unknown='error'`, a descriptive error is returned which maps to a Python `ValueError`.
   - *Custom categories order*: If a custom list of categories is provided, we preserve the user-defined order rather than sorting.

---

## 3. Caveats

- **NumPy Matrix Layouts**: ndarray uses standard row-major (C-style) layout by default. If Python users pass Fortran-style contiguous matrices, standard slicing might trigger copies. We assume and coerce inputs to standard numpy-contiguous layouts in the Python wrappers.
- **Copy vs View**: During transformations, Python wrapper classes perform validation and can copy the arrays where necessary to avoid modifying the user's data in place.
- **Stable ABI Compatibility**: By specifying `abi3-py38` on PyO3, we use the `Bound<'py, ...>` API, which is robust and compatible across Python versions 3.8 to 3.14+.

---

## 4. Conclusion

We recommend the following implementations for the Rust core, PyO3 bindings, and Python wrappers.

### 4.1. Core Rust Implementation (`crates/thermite-core`)

#### `crates/thermite-core/src/model_selection.rs`
```rust
use std::collections::HashMap;
use rand::prelude::*;
use rand::rngs::SmallRng;

pub fn calculate_split_lengths(
    n_samples: usize,
    test_size: Option<f64>,
    train_size: Option<f64>,
) -> Result<(usize, usize), &'static str> {
    let (test_sz, train_sz) = match (test_size, train_size) {
        (None, None) => (Some(0.25), None),
        (t, r) => (t, r),
    };

    let n_test = match test_sz {
        Some(t) => {
            if t < 0.0 {
                return Err("test_size must be non-negative");
            }
            if t >= 1.0 {
                let count = t as usize;
                if count > n_samples {
                    return Err("test_size is greater than n_samples");
                }
                count
            } else {
                let count = (t * n_samples as f64).round() as usize;
                count
            }
        }
        None => 0,
    };

    let n_train = match train_sz {
        Some(r) => {
            if r < 0.0 {
                return Err("train_size must be non-negative");
            }
            if r >= 1.0 {
                let count = r as usize;
                if count > n_samples {
                    return Err("train_size is greater than n_samples");
                }
                count
            } else {
                let count = (r * n_samples as f64).round() as usize;
                count
            }
        }
        None => 0,
    };

    let (final_train, final_test) = match (train_sz, test_sz) {
        (Some(_), Some(_)) => {
            if n_train + n_test > n_samples {
                return Err("The sum of train_size and test_size is greater than n_samples");
            }
            (n_train, n_test)
        }
        (Some(_), None) => {
            (n_train, n_samples - n_train)
        }
        (None, Some(_)) => {
            (n_samples - n_test, n_test)
        }
        (None, None) => unreachable!(),
    };

    if final_train == 0 && final_test == 0 {
        return Err("With n_samples=0, train_size and test_size cannot be resolved");
    }

    Ok((final_train, final_test))
}

pub fn get_split_indices(
    n_samples: usize,
    test_size: Option<f64>,
    train_size: Option<f64>,
    shuffle: bool,
    seed: Option<u64>,
    stratify: Option<&[i64]>,
) -> Result<(Vec<usize>, Vec<usize>), String> {
    let (n_train, n_test) = calculate_split_lengths(n_samples, test_size, train_size)
        .map_err(|e| e.to_string())?;

    let mut rng = match seed {
        Some(s) => SmallRng::seed_from_u64(s),
        None => SmallRng::from_entropy(),
    };

    if let Some(stratify_labels) = stratify {
        if stratify_labels.len() != n_samples {
            return Err("stratify must be of length n_samples".to_string());
        }

        // Group indices by label
        let mut class_to_indices = HashMap::new();
        for (idx, &label) in stratify_labels.iter().enumerate() {
            class_to_indices.entry(label).or_insert_with(Vec::new).push(idx);
        }

        if shuffle {
            for indices in class_to_indices.values_mut() {
                indices.shuffle(&mut rng);
            }
        }

        // Apply Largest Remainder Method (Hamilton method) to distribute test size
        let mut class_test_counts = HashMap::new();
        let mut base_allocated = 0;
        let mut remainders = Vec::new();

        for (&label, indices) in &class_to_indices {
            let nc = indices.len();
            let float_alloc = n_test as f64 * (nc as f64 / n_samples as f64);
            let base_alloc = float_alloc.floor() as usize;
            let rem = float_alloc - base_alloc as f64;
            class_test_counts.insert(label, base_alloc);
            base_allocated += base_alloc;
            remainders.push((label, rem));
        }

        let diff = n_test - base_allocated;
        remainders.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        for i in 0..diff {
            let label = remainders[i].0;
            if let Some(count) = class_test_counts.get_mut(&label) {
                *count += 1;
            }
        }

        let mut train_indices = Vec::new();
        let mut test_indices = Vec::new();

        for (label, indices) in class_to_indices {
            let test_count = *class_test_counts.get(&label).unwrap();
            test_indices.extend_from_slice(&indices[0..test_count]);
            train_indices.extend_from_slice(&indices[test_count..]);
        }

        if shuffle {
            train_indices.shuffle(&mut rng);
            test_indices.shuffle(&mut rng);
        }

        Ok((train_indices, test_indices))
    } else {
        let mut indices: Vec<usize> = (0..n_samples).collect();
        if shuffle {
            indices.shuffle(&mut rng);
        }

        let train_indices = indices[0..n_train].to_vec();
        let test_indices = indices[n_train..(n_train + n_test)].to_vec();
        Ok((train_indices, test_indices))
    }
}
```

#### `crates/thermite-core/src/preprocessing.rs`
```rust
use ndarray::{Array1, Array2, Axis, ArrayView1};
use rayon::prelude::*;
use std::collections::HashMap;

// --- StandardScaler ---

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
        let n_features = x.ncols();
        if n_samples == 0 || n_features == 0 {
            return;
        }

        let columns: Vec<ArrayView1<f64>> = x.axis_iter(Axis(1)).collect();
        let stats: Vec<(f64, f64)> = columns.par_iter().map(|col| {
            let n = col.len() as f64;
            let sum: f64 = col.sum();
            let mean = sum / n;
            let sum_sq_diff: f64 = col.iter().map(|&val| (val - mean).powi(2)).sum();
            let var = sum_sq_diff / n;
            (mean, var)
        }).collect();

        let mut mean_arr = Array1::zeros(n_features);
        let mut var_arr = Array1::zeros(n_features);
        let mut scale_arr = Array1::zeros(n_features);

        for j in 0..n_features {
            let (m, v) = stats[j];
            mean_arr[j] = m;
            var_arr[j] = v;
            let std_dev = v.sqrt();
            scale_arr[j] = if std_dev == 0.0 { 1.0 } else { std_dev };
        }

        if self.with_mean || self.with_std {
            self.mean = Some(mean_arr);
        }
        if self.with_std {
            self.var = Some(var_arr);
            self.scale = Some(scale_arr);
        }
        self.n_samples_seen = n_samples;
    }

    pub fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>, String> {
        let n_features = x.ncols();
        if self.with_mean && self.mean.is_none() {
            return Err("Scaler is not fitted".to_string());
        }
        if self.with_std && self.scale.is_none() {
            return Err("Scaler is not fitted".to_string());
        }

        let mut res = x.clone();
        res.axis_iter_mut(Axis(1)).into_iter().enumerate().par_bridge().for_each(|(j, mut col)| {
            if self.with_mean {
                if let Some(ref mean) = self.mean {
                    let m = mean[j];
                    col.mapv_inplace(|v| v - m);
                }
            }
            if self.with_std {
                if let Some(ref scale) = self.scale {
                    let s = scale[j];
                    col.mapv_inplace(|v| v / s);
                }
            }
        });

        Ok(res)
    }

    pub fn inverse_transform(&self, x: &Array2<f64>) -> Result<Array2<f64>, String> {
        let n_features = x.ncols();
        if self.with_mean && self.mean.is_none() {
            return Err("Scaler is not fitted".to_string());
        }
        if self.with_std && self.scale.is_none() {
            return Err("Scaler is not fitted".to_string());
        }

        let mut res = x.clone();
        res.axis_iter_mut(Axis(1)).into_iter().enumerate().par_bridge().for_each(|(j, mut col)| {
            if self.with_std {
                if let Some(ref scale) = self.scale {
                    let s = scale[j];
                    col.mapv_inplace(|v| v * s);
                }
            }
            if self.with_mean {
                if let Some(ref mean) = self.mean {
                    let m = mean[j];
                    col.mapv_inplace(|v| v + m);
                }
            }
        });

        Ok(res)
    }
}

// --- MinMaxScaler ---

#[derive(Debug, Clone)]
pub struct MinMaxScaler {
    pub feature_range: (f64, f64),
    pub data_min: Option<Array1<f64>>,
    pub data_max: Option<Array1<f64>>,
    pub data_range: Option<Array1<f64>>,
    pub scale: Option<Array1<f64>>,
    pub min: Option<Array1<f64>>,
    pub n_samples_seen: usize,
}

impl MinMaxScaler {
    pub fn new(feature_min: f64, feature_max: f64) -> Self {
        Self {
            feature_range: (feature_min, feature_max),
            data_min: None,
            data_max: None,
            data_range: None,
            scale: None,
            min: None,
            n_samples_seen: 0,
        }
    }

    pub fn fit(&mut self, x: &Array2<f64>) -> Result<(), String> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        if n_samples == 0 || n_features == 0 {
            return Err("Input array is empty".to_string());
        }

        let columns: Vec<ArrayView1<f64>> = x.axis_iter(Axis(1)).collect();
        let stats: Vec<(f64, f64)> = columns.par_iter().map(|col| {
            let mut min_val = f64::INFINITY;
            let mut max_val = f64::NEG_INFINITY;
            for &val in col {
                if val < min_val { min_val = val; }
                if val > max_val { max_val = val; }
            }
            (min_val, max_val)
        }).collect();

        let mut data_min = Array1::zeros(n_features);
        let mut data_max = Array1::zeros(n_features);
        let mut data_range = Array1::zeros(n_features);
        let mut scale = Array1::zeros(n_features);
        let mut min = Array1::zeros(n_features);

        let (range_min, range_max) = self.feature_range;
        let range_diff = range_max - range_min;

        for j in 0..n_features {
            let (col_min, col_max) = stats[j];
            data_min[j] = col_min;
            data_max[j] = col_max;
            let diff = col_max - col_min;
            data_range[j] = diff;

            if diff == 0.0 {
                scale[j] = 0.0;
                min[j] = range_min;
            } else {
                scale[j] = range_diff / diff;
                min[j] = range_min - col_min * scale[j];
            }
        }

        self.data_min = Some(data_min);
        self.data_max = Some(data_max);
        self.data_range = Some(data_range);
        self.scale = Some(scale);
        self.min = Some(min);
        self.n_samples_seen = n_samples;

        Ok(())
    }

    pub fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>, String> {
        let n_features = x.ncols();
        if self.scale.is_none() {
            return Err("MinMaxScaler is not fitted yet".to_string());
        }
        let scale = self.scale.as_ref().unwrap();
        let min = self.min.as_ref().unwrap();

        if n_features != scale.len() {
            return Err(format!("Feature count mismatch: got {}, expected {}", n_features, scale.len()));
        }

        let mut res = x.clone();
        res.axis_iter_mut(Axis(1)).into_iter().enumerate().par_bridge().for_each(|(j, mut col)| {
            let s = scale[j];
            let m = min[j];
            col.mapv_inplace(|v| v * s + m);
        });

        Ok(res)
    }

    pub fn inverse_transform(&self, x: &Array2<f64>) -> Result<Array2<f64>, String> {
        let n_features = x.ncols();
        if self.scale.is_none() {
            return Err("MinMaxScaler is not fitted yet".to_string());
        }
        let scale = self.scale.as_ref().unwrap();
        let min = self.min.as_ref().unwrap();
        let data_min = self.data_min.as_ref().unwrap();

        if n_features != scale.len() {
            return Err(format!("Feature count mismatch: got {}, expected {}", n_features, scale.len()));
        }

        let mut res = x.clone();
        res.axis_iter_mut(Axis(1)).into_iter().enumerate().par_bridge().for_each(|(j, mut col)| {
            let s = scale[j];
            let m = min[j];
            let d_min = data_min[j];
            if s == 0.0 {
                col.fill(d_min);
            } else {
                col.mapv_inplace(|v| (v - m) / s);
            }
        });

        Ok(res)
    }
}

// --- Categorical Value & Encoders ---

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CategoricalValue {
    Int(i64),
    Str(String),
}

impl PartialOrd for CategoricalValue {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CategoricalValue {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self, other) {
            (CategoricalValue::Int(a), CategoricalValue::Int(b)) => a.cmp(b),
            (CategoricalValue::Str(a), CategoricalValue::Str(b)) => a.cmp(b),
            (CategoricalValue::Int(_), CategoricalValue::Str(_)) => std::cmp::Ordering::Less,
            (CategoricalValue::Str(_), CategoricalValue::Int(_)) => std::cmp::Ordering::Greater,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct LabelEncoder {
    pub classes: Vec<CategoricalValue>,
    pub class_to_idx: HashMap<CategoricalValue, usize>,
}

impl LabelEncoder {
    pub fn new() -> Self {
        Self {
            classes: Vec::new(),
            class_to_idx: HashMap::new(),
        }
    }

    pub fn fit(&mut self, y: &[CategoricalValue]) {
        let mut unique: Vec<CategoricalValue> = y.iter().cloned().collect();
        unique.sort();
        unique.dedup();

        let mut class_to_idx = HashMap::new();
        for (idx, class) in unique.iter().enumerate() {
            class_to_idx.insert(class.clone(), idx);
        }

        self.classes = unique;
        self.class_to_idx = class_to_idx;
    }

    pub fn transform(&self, y: &[CategoricalValue]) -> Result<Vec<usize>, String> {
        if self.classes.is_empty() {
            return Err("LabelEncoder is not fitted".to_string());
        }
        let mut encoded = Vec::with_capacity(y.len());
        for item in y {
            if let Some(&idx) = self.class_to_idx.get(item) {
                encoded.push(idx);
            } else {
                return Err(format!("Unseen label: {:?}", item));
            }
        }
        Ok(encoded)
    }

    pub fn inverse_transform(&self, indices: &[usize]) -> Result<Vec<CategoricalValue>, String> {
        if self.classes.is_empty() {
            return Err("LabelEncoder is not fitted".to_string());
        }
        let mut decoded = Vec::with_capacity(indices.len());
        for &idx in indices {
            if idx >= self.classes.len() {
                return Err(format!("Index {} is out of bounds for LabelEncoder", idx));
            }
            decoded.push(self.classes[idx].clone());
        }
        Ok(decoded)
    }
}

#[derive(Debug, Clone)]
pub struct OneHotEncoder {
    pub drop_spec: Option<String>,
    pub handle_unknown: String,
    pub categories: Vec<Vec<CategoricalValue>>,
    pub feature_offsets: Vec<usize>,
    pub total_encoded_cols: usize,
    pub dropped_indices: Vec<Option<usize>>,
}

impl OneHotEncoder {
    pub fn new(drop_spec: Option<String>, handle_unknown: String) -> Self {
        Self {
            drop_spec,
            handle_unknown,
            categories: Vec::new(),
            feature_offsets: Vec::new(),
            total_encoded_cols: 0,
            dropped_indices: Vec::new(),
        }
    }

    pub fn fit(
        &mut self,
        x: &[Vec<CategoricalValue>],
        custom_categories: Option<Vec<Vec<CategoricalValue>>>,
    ) -> Result<(), String> {
        let n_samples = x.len();
        if n_samples == 0 && custom_categories.is_none() {
            return Err("Input array is empty and no custom categories provided".to_string());
        }
        let n_features = if n_samples > 0 { x[0].len() } else { custom_categories.as_ref().unwrap().len() };

        if n_samples > 0 {
            for row in x {
                if row.len() != n_features {
                    return Err("Rows have inconsistent feature sizes".to_string());
                }
            }
        }

        let mut categories = Vec::with_capacity(n_features);
        if let Some(custom) = custom_categories {
            if custom.len() != n_features {
                return Err("Custom categories size does not match feature size".to_string());
            }
            categories = custom;
        } else {
            for col_idx in 0..n_features {
                let mut unique = Vec::new();
                for row in x {
                    unique.push(row[col_idx].clone());
                }
                unique.sort();
                unique.dedup();
                categories.push(unique);
            }
        }

        let mut dropped_indices = vec![None; n_features];
        if let Some(ref drop) = self.drop_spec {
            if drop == "first" {
                for j in 0..n_features {
                    if !categories[j].is_empty() {
                        dropped_indices[j] = Some(0);
                    }
                }
            }
        }

        let mut offsets = Vec::with_capacity(n_features);
        let mut total_cols = 0;
        for (j, col_cats) in categories.iter().enumerate() {
            offsets.push(total_cols);
            let n_cats = col_cats.len();
            let encoded_len = if dropped_indices[j].is_some() {
                if n_cats > 0 { n_cats - 1 } else { 0 }
            } else {
                n_cats
            };
            total_cols += encoded_len;
        }

        self.categories = categories;
        self.feature_offsets = offsets;
        self.total_encoded_cols = total_cols;
        self.dropped_indices = dropped_indices;

        Ok(())
    }

    pub fn transform(&self, x: &[Vec<CategoricalValue>]) -> Result<Array2<f64>, String> {
        let n_samples = x.len();
        if n_samples == 0 {
            return Ok(Array2::zeros((0, self.total_encoded_cols)));
        }
        let n_features = x[0].len();
        if n_features != self.categories.len() {
            return Err("Feature size mismatch during transform".to_string());
        }

        let mut output_flat = vec![0.0; n_samples * self.total_encoded_cols];

        output_flat.par_chunks_mut(self.total_encoded_cols)
            .enumerate()
            .try_for_each(|(i, row_slice)| -> Result<(), String> {
                let sample_row = &x[i];
                for j in 0..n_features {
                    let val = &sample_row[j];
                    let cats = &self.categories[j];

                    if let Some(pos) = cats.iter().position(|c| c == val) {
                        if let Some(drop_idx) = self.dropped_indices[j] {
                            if pos == drop_idx {
                                continue;
                            } else if pos < drop_idx {
                                let out_col = self.feature_offsets[j] + pos;
                                row_slice[out_col] = 1.0;
                            } else {
                                let out_col = self.feature_offsets[j] + pos - 1;
                                row_slice[out_col] = 1.0;
                            }
                        } else {
                            let out_col = self.feature_offsets[j] + pos;
                            row_slice[out_col] = 1.0;
                        }
                    } else {
                        if self.handle_unknown == "ignore" {
                            continue;
                        } else {
                            return Err(format!("Unseen category '{:?}' in feature column {}", val, j));
                        }
                    }
                }
                Ok(())
            })?;

        let arr = Array2::from_shape_vec((n_samples, self.total_encoded_cols), output_flat)
            .map_err(|e| e.to_string())?;
        Ok(arr)
    }

    pub fn inverse_transform(&self, x: &Array2<f64>) -> Result<Vec<Vec<CategoricalValue>>, String> {
        let n_samples = x.nrows();
        let n_cols = x.ncols();
        if n_cols != self.total_encoded_cols {
            return Err("Column count mismatch in OneHotEncoder inverse_transform".to_string());
        }

        let n_features = self.categories.len();
        let mut reconstructed = vec![vec![CategoricalValue::Int(0); n_features]; n_samples];

        reconstructed.par_iter_mut().enumerate().try_for_each(|(i, row)| -> Result<(), String> {
            for j in 0..n_features {
                let cats = &self.categories[j];
                let offset = self.feature_offsets[j];
                let n_cats = cats.len();

                if let Some(drop_idx) = self.dropped_indices[j] {
                    let encoded_len = if n_cats > 0 { n_cats - 1 } else { 0 };
                    if encoded_len == 0 {
                        if n_cats > 0 {
                            row[j] = cats[0].clone();
                        } else {
                            return Err("Category list is empty".to_string());
                        }
                        continue;
                    }

                    let mut max_idx = None;
                    let mut max_val = 0.5;
                    for k in 0..encoded_len {
                        let val = x[[i, offset + k]];
                        if val > max_val {
                            max_val = val;
                            max_idx = Some(k);
                        }
                    }

                    if let Some(idx) = max_idx {
                        if idx < drop_idx {
                            row[j] = cats[idx].clone();
                        } else {
                            row[j] = cats[idx + 1].clone();
                        }
                    } else {
                        row[j] = cats[drop_idx].clone();
                    }
                } else {
                    if n_cats == 0 {
                        return Err("Category list is empty".to_string());
                    }
                    let mut max_idx = 0;
                    let mut max_val = x[[i, offset]];
                    for k in 1..n_cats {
                        let val = x[[i, offset + k]];
                        if val > max_val {
                            max_val = val;
                            max_idx = k;
                        }
                    }
                    row[j] = cats[max_idx].clone();
                }
            }
            Ok(())
        })?;

        Ok(reconstructed)
    }
}
```

---

### 4.2. PyO3 Binding Integration (`crates/thermite-binding`)

We expose these classes and mapping support inside `crates/thermite-binding/src/lib.rs`.

#### `crates/thermite-binding/src/lib.rs`
```rust
use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyReadonlyArray2, ToPyArray};
use thermite_core::preprocessing::{
    StandardScaler as CoreStandardScaler,
    MinMaxScaler as CoreMinMaxScaler,
    LabelEncoder as CoreLabelEncoder,
    OneHotEncoder as CoreOneHotEncoder,
    CategoricalValue,
};
use thermite_core::model_selection::get_split_indices as core_get_split_indices;

// --- PyO3 CategoricalValue converters ---

impl<'source> FromPyObject<'source> for CategoricalValue {
    fn extract_bound(ob: &Bound<'source, PyAny>) -> PyResult<Self> {
        if let Ok(val) = ob.extract::<i64>() {
            Ok(CategoricalValue::Int(val))
        } else if let Ok(val) = ob.extract::<String>() {
            Ok(CategoricalValue::Str(val))
        } else {
            Ok(CategoricalValue::Str(ob.to_string()))
        }
    }
}

impl<'py> IntoPyObject<'py> for CategoricalValue {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self {
            CategoricalValue::Int(val) => val.into_pyobject(py),
            CategoricalValue::Str(val) => val.into_pyobject(py),
        }
    }
}

// --- Bindings: Model Selection ---

#[pyfunction]
#[pyo3(signature = (n_samples, test_size=None, train_size=None, shuffle=true, seed=None, stratify=None))]
fn get_split_indices(
    n_samples: usize,
    test_size: Option<f64>,
    train_size: Option<f64>,
    shuffle: bool,
    seed: Option<u64>,
    stratify: Option<Vec<i64>>,
) -> PyResult<(Vec<usize>, Vec<usize>)> {
    core_get_split_indices(
        n_samples,
        test_size,
        train_size,
        shuffle,
        seed,
        stratify.as_deref(),
    ).map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
}

// --- Bindings: StandardScaler ---

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
        let array = x.as_array();
        self.inner.fit(&array.to_owned());
    }

    fn transform<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let array = x.as_array();
        let result = self.inner.transform(&array.to_owned())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
        Ok(result.to_pyarray_bound(py))
    }

    fn fit_transform<'py>(&mut self, py: Python<'py>, x: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let array = x.as_array();
        let owned = array.to_owned();
        self.inner.fit(&owned);
        let result = self.inner.transform(&owned)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
        Ok(result.to_pyarray_bound(py))
    }

    fn inverse_transform<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let array = x.as_array();
        let result = self.inner.inverse_transform(&array.to_owned())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
        Ok(result.to_pyarray_bound(py))
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
    fn scale<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner.scale.as_ref().map(|s| s.to_pyarray_bound(py))
    }

    #[getter]
    fn n_samples_seen(&self) -> usize {
        self.inner.n_samples_seen
    }
}

// --- Bindings: MinMaxScaler ---

#[pyclass(name = "MinMaxScalerRust")]
pub struct PyMinMaxScaler {
    inner: CoreMinMaxScaler,
}

#[pymethods]
impl PyMinMaxScaler {
    #[new]
    #[pyo3(signature = (feature_min=0.0, feature_max=1.0))]
    fn new(feature_min: f64, feature_max: f64) -> Self {
        Self {
            inner: CoreMinMaxScaler::new(feature_min, feature_max),
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<f64>) -> PyResult<()> {
        let array = x.as_array();
        self.inner.fit(&array.to_owned())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
        Ok(())
    }

    fn transform<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let array = x.as_array();
        let result = self.inner.transform(&array.to_owned())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
        Ok(result.to_pyarray_bound(py))
    }

    fn fit_transform<'py>(&mut self, py: Python<'py>, x: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let array = x.as_array();
        let owned = array.to_owned();
        self.inner.fit(&owned)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
        let result = self.inner.transform(&owned)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
        Ok(result.to_pyarray_bound(py))
    }

    fn inverse_transform<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let array = x.as_array();
        let result = self.inner.inverse_transform(&array.to_owned())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
        Ok(result.to_pyarray_bound(py))
    }

    #[getter]
    fn data_min<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner.data_min.as_ref().map(|v| v.to_pyarray_bound(py))
    }

    #[getter]
    fn data_max<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner.data_max.as_ref().map(|v| v.to_pyarray_bound(py))
    }

    #[getter]
    fn data_range<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner.data_range.as_ref().map(|v| v.to_pyarray_bound(py))
    }

    #[getter]
    fn scale<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner.scale.as_ref().map(|v| v.to_pyarray_bound(py))
    }

    #[getter]
    fn min<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner.min.as_ref().map(|v| v.to_pyarray_bound(py))
    }

    #[getter]
    fn n_samples_seen(&self) -> usize {
        self.inner.n_samples_seen
    }
}

// --- Bindings: LabelEncoder ---

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

    fn fit(&mut self, y: Vec<CategoricalValue>) {
        self.inner.fit(&y);
    }

    fn transform(&self, y: Vec<CategoricalValue>) -> PyResult<Vec<usize>> {
        self.inner.transform(&y).map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
    }

    fn fit_transform(&mut self, y: Vec<CategoricalValue>) -> PyResult<Vec<usize>> {
        self.inner.fit(&y);
        self.inner.transform(&y).map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
    }

    fn inverse_transform(&self, indices: Vec<usize>) -> PyResult<Vec<CategoricalValue>> {
        self.inner.inverse_transform(&indices).map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
    }

    fn classes(&self) -> Vec<CategoricalValue> {
        self.inner.classes.clone()
    }
}

// --- Bindings: OneHotEncoder ---

#[pyclass(name = "OneHotEncoderRust")]
pub struct PyOneHotEncoder {
    inner: CoreOneHotEncoder,
}

#[pymethods]
impl PyOneHotEncoder {
    #[new]
    #[pyo3(signature = (drop=None, handle_unknown=String::from("error")))]
    fn new(drop: Option<String>, handle_unknown: String) -> PyResult<Self> {
        if handle_unknown != "error" && handle_unknown != "ignore" {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "handle_unknown must be 'error' or 'ignore'",
            ));
        }
        Ok(Self {
            inner: CoreOneHotEncoder::new(drop, handle_unknown),
        }
    }

    fn fit(&mut self, x: Vec<Vec<CategoricalValue>>, custom_categories: Option<Vec<Vec<CategoricalValue>>>) -> PyResult<()> {
        self.inner.fit(&x, custom_categories).map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
        Ok(())
    }

    fn transform<'py>(&self, py: Python<'py>, x: Vec<Vec<CategoricalValue>>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let result = self.inner.transform(&x).map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
        Ok(result.to_pyarray_bound(py))
    }

    fn fit_transform<'py>(&mut self, py: Python<'py>, x: Vec<Vec<CategoricalValue>>, custom_categories: Option<Vec<Vec<CategoricalValue>>>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.inner.fit(&x, custom_categories).map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
        let result = self.inner.transform(&x).map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
        Ok(result.to_pyarray_bound(py))
    }

    fn inverse_transform(&self, x: PyReadonlyArray2<f64>) -> PyResult<Vec<Vec<CategoricalValue>>> {
        let array = x.as_array();
        self.inner.inverse_transform(&array.to_owned()).map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
    }

    #[getter]
    fn categories(&self) -> Vec<Vec<CategoricalValue>> {
        self.inner.categories.clone()
    }
}

// --- Module Definition ---

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_split_indices, m)?)?;
    m.add_class::<PyStandardScaler>()?;
    m.add_class::<PyMinMaxScaler>()?;
    m.add_class::<PyLabelEncoder>()?;
    m.add_class::<PyOneHotEncoder>()?;
    Ok(())
}
```

---

### 4.3. Python Wrapper Classes (`thermite`)

#### `thermite/model_selection.py`
```python
import numpy as np
from ._core import get_split_indices

def train_test_split(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None):
    if not arrays:
        raise ValueError("At least one array required as input")
    
    np_arrays = [np.asarray(arr) for arr in arrays]
    n_samples = len(np_arrays[0])
    
    for arr in np_arrays:
        if len(arr) != n_samples:
            raise ValueError("Found input variables with inconsistent numbers of samples")
            
    stratify_labels = None
    if stratify is not None:
        stratify_arr = np.asarray(stratify)
        if len(stratify_arr) != n_samples:
            raise ValueError("stratify must have the same length as input arrays")
        # Factorize stratify array to numeric labels
        _, stratify_labels = np.unique(stratify_arr, return_inverse=True)
        stratify_labels = list(stratify_labels.astype(np.int64))

    # Seed extraction
    seed_val = None
    if random_state is not None:
        if isinstance(random_state, (int, np.integer)):
            seed_val = int(random_state)
        elif hasattr(random_state, "randint"):
            # If it's a RandomState/Generator, pull an integer seed
            seed_val = int(random_state.randint(0, 2**32 - 1))
        else:
            raise ValueError("random_state must be an integer, RandomState, or None")

    # Call private PyO3 backend function
    train_idx, test_idx = get_split_indices(
        n_samples=n_samples,
        test_size=test_size,
        train_size=train_size,
        shuffle=shuffle,
        seed=seed_val,
        stratify=stratify_labels
    )
    
    train_idx = np.asarray(train_idx)
    test_idx = np.asarray(test_idx)
    
    res = []
    for arr in np_arrays:
        res.append(arr[train_idx])
        res.append(arr[test_idx])
        
    return res
```

#### `thermite/preprocessing.py`
```python
import numpy as np
from ._core import (
    StandardScalerRust,
    MinMaxScalerRust,
    LabelEncoderRust,
    OneHotEncoderRust,
)

# --- StandardScaler ---

class StandardScaler:
    def __init__(self, *, copy=True, with_mean=True, with_std=True):
        self.copy = copy
        self.with_mean = with_mean
        self.with_std = with_std
        self._inner = StandardScalerRust(with_mean=with_mean, with_std=with_std)

    def fit(self, X, y=None):
        X = self._validate_data(X)
        self._inner.fit(X)
        return self

    def transform(self, X, copy=None):
        X = self._validate_data(X, reset=False)
        return self._inner.transform(X)

    def fit_transform(self, X, y=None, **fit_params):
        X = self._validate_data(X)
        return self._inner.fit_transform(X)

    def inverse_transform(self, X, copy=None):
        X = self._validate_data(X, reset=False)
        return self._inner.inverse_transform(X)

    @property
    def mean_(self):
        return self._inner.mean

    @property
    def var_(self):
        return self._inner.var

    @property
    def scale_(self):
        return self._inner.scale

    @property
    def n_samples_seen_(self):
        return self._inner.n_samples_seen

    def _validate_data(self, X, reset=True):
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2:
            raise ValueError("Expected 2D array, got 1D or scalar instead")
        if X_arr.shape[0] == 0 or X_arr.shape[1] == 0:
            raise ValueError("Empty arrays not supported")
        return X_arr


# --- MinMaxScaler ---

class MinMaxScaler:
    def __init__(self, feature_range=(0, 1), *, copy=True, clip=False):
        self.feature_range = feature_range
        self.copy = copy
        self.clip = clip
        if feature_range[0] >= feature_range[1]:
            raise ValueError("Minimum of feature_range must be smaller than maximum")
        self._inner = MinMaxScalerRust(feature_range[0], feature_range[1])

    def fit(self, X, y=None):
        X = self._validate_data(X)
        self._inner.fit(X)
        return self

    def transform(self, X):
        X = self._validate_data(X, reset=False)
        res = self._inner.transform(X)
        if self.clip:
            res = np.clip(res, self.feature_range[0], self.feature_range[1])
        return res

    def fit_transform(self, X, y=None, **fit_params):
        X = self._validate_data(X)
        res = self._inner.fit_transform(X)
        if self.clip:
            res = np.clip(res, self.feature_range[0], self.feature_range[1])
        return res

    def inverse_transform(self, X):
        X = self._validate_data(X, reset=False)
        return self._inner.inverse_transform(X)

    @property
    def data_min_(self):
        return self._inner.data_min

    @property
    def data_max_(self):
        return self._inner.data_max

    @property
    def data_range_(self):
        return self._inner.data_range

    @property
    def scale_(self):
        return self._inner.scale

    @property
    def min_(self):
        return self._inner.min

    @property
    def n_samples_seen_(self):
        return self._inner.n_samples_seen

    def _validate_data(self, X, reset=True):
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2:
            raise ValueError("Expected 2D array, got 1D or scalar instead")
        if X_arr.shape[0] == 0 or X_arr.shape[1] == 0:
            raise ValueError("Empty arrays not supported")
        return X_arr


# --- LabelEncoder ---

class LabelEncoder:
    def __init__(self):
        self._inner = LabelEncoderRust()
        self.classes_ = None
        self._dtype = None

    def fit(self, y):
        y_arr = np.asarray(y)
        if y_arr.ndim != 1:
            raise ValueError("Expected 1D array")
        self._dtype = y_arr.dtype
        # Convert array to a list of python strings or ints
        y_list = y_arr.tolist()
        self._inner.fit(y_list)
        # Parse classes back to original dtype if numeric
        classes_raw = self._inner.classes()
        if np.issubdtype(self._dtype, np.integer):
            self.classes_ = np.array([int(x) for x in classes_raw], dtype=self._dtype)
        elif np.issubdtype(self._dtype, np.floating):
            self.classes_ = np.array([float(x) for x in classes_raw], dtype=self._dtype)
        else:
            self.classes_ = np.array(classes_raw, dtype=self._dtype)
        return self

    def transform(self, y):
        if self.classes_ is None:
            raise ValueError("LabelEncoder is not fitted")
        y_arr = np.asarray(y)
        y_list = y_arr.tolist()
        res = self._inner.transform(y_list)
        return np.array(res, dtype=np.int64)

    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)

    def inverse_transform(self, y):
        if self.classes_ is None:
            raise ValueError("LabelEncoder is not fitted")
        y_arr = np.asarray(y, dtype=np.int64)
        indices = y_arr.tolist()
        res_raw = self._inner.inverse_transform(indices)
        if np.issubdtype(self._dtype, np.integer):
            return np.array([int(x) for x in res_raw], dtype=self._dtype)
        elif np.issubdtype(self._dtype, np.floating):
            return np.array([float(x) for x in res_raw], dtype=self._dtype)
        else:
            return np.array(res_raw, dtype=self._dtype)


# --- OneHotEncoder ---

class OneHotEncoder:
    def __init__(self, *, categories="auto", drop=None, sparse_output=False, handle_unknown="error"):
        if sparse_output:
            raise ValueError("Sparse output is not supported yet in Thermite OneHotEncoder")
        if handle_unknown not in ("error", "ignore"):
            raise ValueError("handle_unknown must be 'error' or 'ignore'")
            
        self.categories = categories
        self.drop = drop
        self.sparse_output = sparse_output
        self.handle_unknown = handle_unknown
        
        drop_spec = "first" if drop == "first" else None
        self._inner = OneHotEncoderRust(drop=drop_spec, handle_unknown=handle_unknown)
        self.categories_ = None

    def fit(self, X, y=None):
        X = self._validate_data(X)
        X_list = X.tolist()
        
        custom_cats = None
        if self.categories != "auto":
            custom_cats = [list(cat) for cat in self.categories]
            
        self._inner.fit(X_list, custom_cats)
        self._set_fitted_categories(X.dtype)
        return self

    def transform(self, X):
        if self.categories_ is None:
            raise ValueError("OneHotEncoder is not fitted")
        X = self._validate_data(X)
        X_list = X.tolist()
        return self._inner.transform(X_list)

    def fit_transform(self, X, y=None, **fit_params):
        X = self._validate_data(X)
        X_list = X.tolist()
        
        custom_cats = None
        if self.categories != "auto":
            custom_cats = [list(cat) for cat in self.categories]
            
        res = self._inner.fit_transform(X_list, custom_cats)
        self._set_fitted_categories(X.dtype)
        return res

    def inverse_transform(self, X):
        if self.categories_ is None:
            raise ValueError("OneHotEncoder is not fitted")
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2:
            raise ValueError("Expected 2D array for inverse_transform")
            
        res_list = self._inner.inverse_transform(X_arr)
        
        # Cast to object arrays to preserve types (mixed strings/ints)
        return np.array(res_list, dtype=object)

    def _set_fitted_categories(self, dtype):
        cats_raw = self._inner.categories
        self.categories_ = []
        for cat in cats_raw:
            if np.issubdtype(dtype, np.integer):
                self.categories_.append(np.array([int(x) for x in cat], dtype=dtype))
            elif np.issubdtype(dtype, np.floating):
                self.categories_.append(np.array([float(x) for x in cat], dtype=dtype))
            else:
                self.categories_.append(np.array(cat, dtype=dtype))

    def _validate_data(self, X):
        X_arr = np.asarray(X)
        if X_arr.ndim != 2:
            raise ValueError("Expected 2D array, got 1D or scalar instead")
        if X_arr.shape[0] == 0 or X_arr.shape[1] == 0:
            raise ValueError("Empty arrays not supported")
        return X_arr
```

---

## 5. Verification Method

To independently verify the implementation after code files are populated:

1. **Rust Core Tests**:
   Create unit tests in `crates/thermite-core/src/preprocessing.rs` (in a `mod tests`) and `crates/thermite-core/src/model_selection.rs` targeting the basic functionality of the split calculations and scaler calculations.
   Command:
   ```bash
   cargo test
   ```
   *Expected outcome*: All unit tests compile and pass.

2. **Maturin Build**:
   Verify compilation and python package generation:
   ```bash
   maturin develop
   ```
   *Expected outcome*: Rust modules compile cleanly, private extension `thermite._core` is generated, and editable packaging is installed.

3. **E2E Integration & Verification Suite**:
   Run the full set of preprocessing and model selection tests:
   ```bash
   PYTHONPATH=. pytest tests/test_tier1_preprocessing.py tests/test_tier1_model_selection_pipeline.py
   ```
   *Expected outcome*: All tests pass against the new Thermite backend.
