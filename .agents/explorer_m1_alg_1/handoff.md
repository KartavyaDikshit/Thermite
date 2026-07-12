# Handoff Report  Algorithm & Preprocessing Design (M1-2 & M1-3)

This report details the architectural designs, Rust implementation templates, and PyO3 dynamic bindings for `train_test_split`, `StandardScaler`, `MinMaxScaler`, `LabelEncoder`, and `OneHotEncoder`.

---

## 1. Observation
We observed and analyzed the workspace structure:
* The build layout and workspace dependencies are configured in root `Cargo.toml` with `ndarray = "0.15"`, `rayon = "1.8"`, `rand = "0.8"`, `pyo3 = "0.21"`, and `numpy = "0.21"`.
* Code modules:
  * `/Users/kartavyadikshit/Projects/Thermite/crates/thermite-core/src/lib.rs` exports `preprocessing` and `model_selection`.
  * `/Users/kartavyadikshit/Projects/Thermite/crates/thermite-binding/src/lib.rs` registers PyO3 symbols.
  * `/Users/kartavyadikshit/Projects/Thermite/thermite/preprocessing.py` and `/Users/kartavyadikshit/Projects/Thermite/thermite/model_selection.py` are empty Python wrappers ready to wrap the binary module `thermite._core`.
* Python arrays can contain multi-dimensional numerical values, integers, or strings. Slicing/fancy indexing inside Rust for generic dtypes is extremely verbose, but PyO3 can delegate slice/index lookup to NumPy via Python runtime wrappers or Rust helper index-arrays.

---

## 2. Logic Chain
1. **Generic Array Types in train_test_split**: The user requires `train_test_split` to work with multiple arbitrary arrays (`*arrays`). Rather than rewriting type-dispatch macros in Rust for every possible NumPy dtype (f64, f32, int64, int32, string, object), the core Rust function `compute_split_indices` can compute and return train/test integer index vectors (`Vec<usize>`). The PyO3 wrapper can then call the dynamic indexing method (`__getitem__`) on the original Python array object. This maintains perfect type, dtype, and shape invariance for all arrays, while keeping the heavy splitting logic (random shuffling and stratified grouping) highly optimized in Rust.
2. **Reproducible SmallRng Selection**: Using `SmallRng` from `rand` with a `u64` seed ensures reproducibility of the splitting sequence. If no seed is provided (`random_state=None`), we instantiate it from entropy (`SmallRng::from_entropy()`).
3. **Largest Remainder Method for Stratification**: To perform stratified splitting:
   * First, group index arrays by labels (represented as strings for universality).
   * Calculate exact allocation sizes per group (e.g. `n_class * test_ratio`).
   * Allocate the floor of the exact size to each class.
   * Sort the remainder fractions descending, and distribute the remaining slots to classes with the highest fractions (Largest Remainder Method). This guarantees that the final split size is exactly equal to the target size, and classes are represented proportionally.
4. **Rayon Column-Wise Reduction**: For `StandardScaler` and `MinMaxScaler`, statistics (mean, variance, min, max) are column-wise independent. Thus, we iterate over columns (`X.axis_iter(Axis(1))`) and use Rayon's parallel iterators (`into_par_iter()`) to compute these statistics in parallel. For `transform` and `inverse_transform`, each row is independent, so we parallelize over rows (`X.axis_iter_mut(Axis(0))`) using `into_par_iter()`.
5. **Categorical Handling in LabelEncoder & OneHotEncoder**:
   * For `LabelEncoder`, we use `Vec<i64>` and `Vec<String>` to support integer and string target categories. Shifting type-casting to the Python wrapper simplifies the Rust binding and avoids complex PyO3 object conversions.
   * `OneHotEncoder` stores an enumeration `ColumnCategories` containing either `Int(Vec<i64>)` or `Str(Vec<String>)` for each feature column. Using binary search over sorted category lists in Rust makes transform checks $O(\log C_j)$ per feature value. Rayon's `try_for_each` is used to parallelize the transformation row-wise while ensuring that unknown categories propagate errors immediately.

---

## 3. Caveats
* **NaNs/Infs**: Under standard configurations, scikit-learn throws errors on NaNs during scaling. We check for finite numbers in Rust and raise `ValueError` from the bindings.
* **Large String Copying**: For string categories, Rust duplicates values during sorting/deduplication. This is extremely safe and has negligible impact, since category vocabularies are typically small.
* **Drop Parameters**: For the MVP, `OneHotEncoder` assumes `drop=None` (no dropped categories). Expanding this later only requires skipping index 0 or a specified category in the transformation offset calculations.

---

## 4. Conclusion & Templates
We have designed the complete implementation layout. Below are the recommended template structures for the files in the workspace.

### A. Core Rust Implementations (`crates/thermite-core`)

#### File: `crates/thermite-core/src/model_selection.rs`
```rust
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct SplitIndices {
    pub train_indices: Vec<usize>,
    pub test_indices: Vec<usize>,
}

pub fn compute_split_indices(
    n_samples: usize,
    test_size: Option<f64>,
    train_size: Option<f64>,
    shuffle: bool,
    random_state: Option<u64>,
    stratify: Option<&[String]>,
) -> Result<SplitIndices, String> {
    if n_samples == 0 {
        return Err("Number of samples must be greater than 0".to_string());
    }

    let (n_train, n_test) = determine_split_sizes(n_samples, test_size, train_size)?;
    let mut indices: Vec<usize> = (0..n_samples).collect();

    let mut rng = match random_state {
        Some(seed) => SmallRng::seed_from_u64(seed),
        None => SmallRng::from_entropy(),
    };

    if let Some(stratify_labels) = stratify {
        if stratify_labels.len() != n_samples {
            return Err("Stratify labels length must match number of samples".to_string());
        }

        // Group indices by label
        let mut label_to_indices: HashMap<String, Vec<usize>> = HashMap::new();
        for (idx, label) in stratify_labels.iter().enumerate() {
            label_to_indices.entry(label.clone()).or_default().push(idx);
        }

        // Shuffle each group
        if shuffle {
            for indices_group in label_to_indices.values_mut() {
                indices_group.shuffle(&mut rng);
            }
        }

        let test_ratio = n_test as f64 / n_samples as f64;
        let mut train_indices = Vec::with_capacity(n_train);
        let mut test_indices = Vec::with_capacity(n_test);

        // Basic allocation using Largest Remainder Method
        let mut class_splits = Vec::new();
        let mut allocated_test = 0;

        for (label, group_idx) in label_to_indices.iter() {
            let n_c = group_idx.len();
            let test_c_exact = n_c as f64 * test_ratio;
            let test_c = test_c_exact.floor() as usize;
            let fract = test_c_exact - test_c as f64;
            class_splits.push((label.clone(), n_c, test_c, fract, group_idx));
            allocated_test += test_c;
        }

        // Distribute remaining test slots
        if allocated_test < n_test {
            class_splits.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());
            let mut diff = n_test - allocated_test;
            for split in class_splits.iter_mut() {
                if diff == 0 {
                    break;
                }
                let max_allocatable = split.1 - split.2;
                if max_allocatable > 0 {
                    split.2 += 1;
                    diff -= 1;
                }
            }
        }

        // Collect train and test indices
        for split in class_splits {
            let group_idx = split.4;
            let test_count = split.2;
            let (test_part, train_part) = group_idx.split_at(test_count);
            test_indices.extend_from_slice(test_part);
            train_indices.extend_from_slice(train_part);
        }

        if shuffle {
            train_indices.shuffle(&mut rng);
            test_indices.shuffle(&mut rng);
        }

        Ok(SplitIndices {
            train_indices,
            test_indices,
        })
    } else {
        if shuffle {
            indices.shuffle(&mut rng);
        }
        let (test_part, train_part) = indices.split_at(n_test);
        Ok(SplitIndices {
            train_indices: train_part.to_vec(),
            test_indices: test_part.to_vec(),
        })
    }
}

fn determine_split_sizes(
    n_samples: usize,
    test_size: Option<f64>,
    train_size: Option<f64>,
) -> Result<(usize, usize), String> {
    let default_test_size = 0.25;

    let (n_train, n_test) = match (test_size, train_size) {
        (None, None) => {
            let n_test = (n_samples as f64 * default_test_size).round() as usize;
            let n_train = n_samples - n_test;
            (n_train, n_test)
        }
        (Some(ts), None) => {
            let n_test = if ts >= 1.0 {
                ts as usize
            } else if ts > 0.0 && ts < 1.0 {
                (n_samples as f64 * ts).round() as usize
            } else {
                return Err("test_size must be > 0".to_string());
            };
            if n_test > n_samples {
                return Err(format!("test_size={} is greater than n_samples={}", n_test, n_samples));
            }
            (n_samples - n_test, n_test)
        }
        (None, Some(tr)) => {
            let n_train = if tr >= 1.0 {
                tr as usize
            } else if tr > 0.0 && tr < 1.0 {
                (n_samples as f64 * tr).round() as usize
            } else {
                return Err("train_size must be > 0".to_string());
            };
            if n_train > n_samples {
                return Err(format!("train_size={} is greater than n_samples={}", n_train, n_samples));
            }
            (n_train, n_samples - n_train)
        }
        (Some(ts), Some(tr)) => {
            let n_test = if ts >= 1.0 {
                ts as usize
            } else if ts > 0.0 && ts < 1.0 {
                (n_samples as f64 * ts).round() as usize
            } else {
                return Err("test_size must be > 0".to_string());
            };
            let n_train = if tr >= 1.0 {
                tr as usize
            } else if tr > 0.0 && tr < 1.0 {
                (n_samples as f64 * tr).round() as usize
            } else {
                return Err("train_size must be > 0".to_string());
            };
            if n_test + n_train > n_samples {
                return Err(format!(
                    "The sum of train_size={} and test_size={} is larger than n_samples={}",
                    n_train, n_test, n_samples
                ));
            }
            (n_train, n_test)
        }
    };

    if n_train == 0 || n_test == 0 {
        return Err("Train or test size cannot be zero".to_string());
    }

    Ok((n_train, n_test))
}
```

#### File: `crates/thermite-core/src/preprocessing.rs`
```rust
use ndarray::{Array1, ArrayView2, Axis};
use rayon::prelude::*;

// ==========================================
// StandardScaler
// ==========================================
pub struct StandardScaler {
    pub mean: Option<Array1<f64>>,
    pub var: Option<Array1<f64>>,
    pub scale: Option<Array1<f64>>,
    pub n_samples_seen: usize,
    pub with_mean: bool,
    pub with_std: bool,
}

impl StandardScaler {
    pub fn new(with_mean: bool, with_std: bool) -> Self {
        StandardScaler {
            mean: None,
            var: None,
            scale: None,
            n_samples_seen: 0,
            with_mean,
            with_std,
        }
    }

    pub fn fit(&mut self, X: &ArrayView2<f64>) -> Result<(), String> {
        if X.is_empty() {
            return Err("Input array is empty".to_string());
        }
        let n_samples = X.nrows();
        let n_features = X.ncols();

        if X.iter().any(|&val| !val.is_finite()) {
            return Err("Input contains NaN or infinity values".to_string());
        }

        // Parallel column-wise statistics
        let stats: Vec<(f64, f64)> = X.axis_iter(Axis(1))
            .into_par_iter()
            .map(|col| {
                let mean = col.mean().unwrap_or(0.0);
                let var = col.variance(0.0); // ddof=0
                (mean, var)
            })
            .collect();

        let mut mean_arr = Array1::zeros(n_features);
        let mut var_arr = Array1::zeros(n_features);
        let mut scale_arr = Array1::zeros(n_features);

        for (i, &(m, v)) in stats.iter().enumerate() {
            mean_arr[i] = m;
            var_arr[i] = v;
            scale_arr[i] = if v == 0.0 { 1.0 } else { v.sqrt() };
        }

        self.mean = Some(mean_arr);
        self.var = Some(var_arr);
        self.scale = Some(scale_arr);
        self.n_samples_seen = n_samples;

        Ok(())
    }

    pub fn transform(&self, X: &ArrayView2<f64>) -> Result<ndarray::Array2<f64>, String> {
        if self.mean.is_none() || self.scale.is_none() {
            return Err("StandardScaler is not fitted yet".to_string());
        }
        let n_features = X.ncols();
        let mean = self.mean.as_ref().unwrap();
        let scale = self.scale.as_ref().unwrap();

        if n_features != mean.len() {
            return Err(format!("Feature mismatch: expected {}, got {}", mean.len(), n_features));
        }

        if X.iter().any(|&val| !val.is_finite()) {
            return Err("Input contains NaN or infinity values".to_string());
        }

        let mut X_transformed = X.to_owned();

        X_transformed.axis_iter_mut(Axis(0))
            .into_par_iter()
            .for_each(|mut row| {
                for j in 0..row.len() {
                    let mut val = row[j];
                    if self.with_mean {
                        val -= mean[j];
                    }
                    if self.with_std {
                        val /= scale[j];
                    }
                    row[j] = val;
                }
            });

        Ok(X_transformed)
    }

    pub fn inverse_transform(&self, X: &ArrayView2<f64>) -> Result<ndarray::Array2<f64>, String> {
        if self.mean.is_none() || self.scale.is_none() {
            return Err("StandardScaler is not fitted yet".to_string());
        }
        let n_features = X.ncols();
        let mean = self.mean.as_ref().unwrap();
        let scale = self.scale.as_ref().unwrap();

        if n_features != mean.len() {
            return Err(format!("Feature mismatch: expected {}, got {}", mean.len(), n_features));
        }

        if X.iter().any(|&val| !val.is_finite()) {
            return Err("Input contains NaN or infinity values".to_string());
        }

        let mut X_orig = X.to_owned();

        X_orig.axis_iter_mut(Axis(0))
            .into_par_iter()
            .for_each(|mut row| {
                for j in 0..row.len() {
                    let mut val = row[j];
                    if self.with_std {
                        val *= scale[j];
                    }
                    if self.with_mean {
                        val += mean[j];
                    }
                    row[j] = val;
                }
            });

        Ok(X_orig)
    }
}

// ==========================================
// MinMaxScaler
// ==========================================
pub struct MinMaxScaler {
    pub data_min: Option<Array1<f64>>,
    pub data_max: Option<Array1<f64>>,
    pub scale: Option<Array1<f64>>,
    pub min: Option<Array1<f64>>,
    pub feature_range: (f64, f64),
}

impl MinMaxScaler {
    pub fn new(feature_range: (f64, f64)) -> Result<Self, String> {
        if feature_range.0 >= feature_range.1 {
            return Err("feature_range min must be less than max".to_string());
        }
        Ok(MinMaxScaler {
            data_min: None,
            data_max: None,
            scale: None,
            min: None,
            feature_range,
        })
    }

    pub fn fit(&mut self, X: &ArrayView2<f64>) -> Result<(), String> {
        if X.is_empty() {
            return Err("Input array is empty".to_string());
        }
        let n_features = X.ncols();

        if X.iter().any(|&val| !val.is_finite()) {
            return Err("Input contains NaN or infinity values".to_string());
        }

        let stats: Vec<(f64, f64)> = X.axis_iter(Axis(1))
            .into_par_iter()
            .map(|col| {
                let mut c_min = f64::INFINITY;
                let mut c_max = f64::NEG_INFINITY;
                for &val in col {
                    if val < c_min { c_min = val; }
                    if val > c_max { c_max = val; }
                }
                (c_min, c_max)
            })
            .collect();

        let mut data_min_arr = Array1::zeros(n_features);
        let mut data_max_arr = Array1::zeros(n_features);
        let mut scale_arr = Array1::zeros(n_features);
        let mut min_arr = Array1::zeros(n_features);

        let (min_val, max_val) = self.feature_range;
        let range_diff = max_val - min_val;

        for (i, &(d_min, d_max)) in stats.iter().enumerate() {
            data_min_arr[i] = d_min;
            data_max_arr[i] = d_max;
            let diff = d_max - d_min;
            if diff == 0.0 {
                scale_arr[i] = 0.0;
                min_arr[i] = min_val;
            } else {
                scale_arr[i] = range_diff / diff;
                min_arr[i] = min_val - d_min * scale_arr[i];
            }
        }

        self.data_min = Some(data_min_arr);
        self.data_max = Some(data_max_arr);
        self.scale = Some(scale_arr);
        self.min = Some(min_arr);

        Ok(())
    }

    pub fn transform(&self, X: &ArrayView2<f64>) -> Result<ndarray::Array2<f64>, String> {
        if self.scale.is_none() || self.min.is_none() {
            return Err("MinMaxScaler is not fitted yet".to_string());
        }
        let n_features = X.ncols();
        let scale = self.scale.as_ref().unwrap();
        let min = self.min.as_ref().unwrap();

        if n_features != scale.len() {
            return Err(format!("Feature mismatch: expected {}, got {}", scale.len(), n_features));
        }

        if X.iter().any(|&val| !val.is_finite()) {
            return Err("Input contains NaN or infinity values".to_string());
        }

        let mut X_transformed = X.to_owned();

        X_transformed.axis_iter_mut(Axis(0))
            .into_par_iter()
            .for_each(|mut row| {
                for j in 0..row.len() {
                    row[j] = row[j] * scale[j] + min[j];
                }
            });

        Ok(X_transformed)
    }

    pub fn inverse_transform(&self, X: &ArrayView2<f64>) -> Result<ndarray::Array2<f64>, String> {
        if self.scale.is_none() || self.min.is_none() || self.data_min.is_none() {
            return Err("MinMaxScaler is not fitted yet".to_string());
        }
        let n_features = X.ncols();
        let scale = self.scale.as_ref().unwrap();
        let min = self.min.as_ref().unwrap();
        let data_min = self.data_min.as_ref().unwrap();

        if n_features != scale.len() {
            return Err(format!("Feature mismatch: expected {}, got {}", scale.len(), n_features));
        }

        if X.iter().any(|&val| !val.is_finite()) {
            return Err("Input contains NaN or infinity values".to_string());
        }

        let mut X_orig = X.to_owned();

        X_orig.axis_iter_mut(Axis(0))
            .into_par_iter()
            .for_each(|mut row| {
                for j in 0..row.len() {
                    if scale[j] == 0.0 {
                        row[j] = data_min[j];
                    } else {
                        row[j] = (row[j] - min[j]) / scale[j];
                    }
                }
            });

        Ok(X_orig)
    }
}

// ==========================================
// LabelEncoder
// ==========================================
pub struct LabelEncoderCore {
    pub classes_int: Option<Vec<i64>>,
    pub classes_str: Option<Vec<String>>,
}

impl LabelEncoderCore {
    pub fn new() -> Self {
        LabelEncoderCore {
            classes_int: None,
            classes_str: None,
        }
    }

    pub fn fit_int(&mut self, y: &[i64]) {
        let mut sorted = y.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        self.classes_int = Some(sorted);
        self.classes_str = None;
    }

    pub fn fit_str(&mut self, y: Vec<String>) {
        let mut sorted = y;
        sorted.sort_unstable();
        sorted.dedup();
        self.classes_str = Some(sorted);
        self.classes_int = None;
    }

    pub fn transform_int(&self, y: &[i64]) -> Result<Vec<i64>, String> {
        let classes = self.classes_int.as_ref().ok_or("LabelEncoder not fitted on integers")?;
        let mut out = Vec::with_capacity(y.len());
        for &val in y {
            match classes.binary_search(&val) {
                Ok(idx) => out.push(idx as i64),
                Err(_) => return Err(format!("y contains previously unseen labels: {}", val)),
            }
        }
        Ok(out)
    }

    pub fn transform_str(&self, y: &[String]) -> Result<Vec<i64>, String> {
        let classes = self.classes_str.as_ref().ok_or("LabelEncoder not fitted on strings")?;
        let mut out = Vec::with_capacity(y.len());
        for val in y {
            match classes.binary_search(val) {
                Ok(idx) => out.push(idx as i64),
                Err(_) => return Err(format!("y contains previously unseen labels: {}", val)),
            }
        }
        Ok(out)
    }

    pub fn inverse_transform_int(&self, y: &[i64]) -> Result<Vec<i64>, String> {
        let classes = self.classes_int.as_ref().ok_or("LabelEncoder not fitted on integers")?;
        let mut out = Vec::with_capacity(y.len());
        for &idx in y {
            if idx < 0 || idx >= classes.len() as i64 {
                return Err(format!("y contains out-of-bounds index: {}", idx));
            }
            out.push(classes[idx as usize]);
        }
        Ok(out)
    }

    pub fn inverse_transform_str(&self, y: &[i64]) -> Result<Vec<String>, String> {
        let classes = self.classes_str.as_ref().ok_or("LabelEncoder not fitted on strings")?;
        let mut out = Vec::with_capacity(y.len());
        for &idx in y {
            if idx < 0 || idx >= classes.len() as i64 {
                return Err(format!("y contains out-of-bounds index: {}", idx));
            }
            out.push(classes[idx as usize].clone());
        }
        Ok(out)
    }
}

// ==========================================
// OneHotEncoder
// ==========================================
pub enum ColumnCategories {
    Int(Vec<i64>),
    Str(Vec<String>),
}

impl ColumnCategories {
    pub fn len(&self) -> usize {
        match self {
            ColumnCategories::Int(v) => v.len(),
            ColumnCategories::Str(v) => v.len(),
        }
    }
}

pub struct OneHotEncoderCore {
    pub categories: Vec<ColumnCategories>,
    pub handle_unknown: String,
}

impl OneHotEncoderCore {
    pub fn new(handle_unknown: String) -> Self {
        OneHotEncoderCore {
            categories: Vec::new(),
            handle_unknown,
        }
    }

    pub fn fit_int(&mut self, X: &ArrayView2<i64>) {
        let n_features = X.ncols();
        let mut categories = Vec::with_capacity(n_features);
        for j in 0..n_features {
            let col = X.column(j);
            let mut col_vec = col.to_vec();
            col_vec.sort_unstable();
            col_vec.dedup();
            categories.push(ColumnCategories::Int(col_vec));
        }
        self.categories = categories;
    }

    pub fn fit_str(&mut self, X_cols: Vec<Vec<String>>) {
        let mut categories = Vec::with_capacity(X_cols.len());
        for col in X_cols {
            let mut col_vec = col;
            col_vec.sort_unstable();
            col_vec.dedup();
            categories.push(ColumnCategories::Str(col_vec));
        }
        self.categories = categories;
    }

    pub fn transform_int(&self, X: &ArrayView2<i64>) -> Result<ndarray::Array2<f64>, String> {
        let n_samples = X.nrows();
        let n_features = X.ncols();

        if n_features != self.categories.len() {
            return Err(format!("Feature mismatch: expected {}, got {}", self.categories.len(), n_features));
        }

        let total_categories: usize = self.categories.iter().map(|c| c.len()).sum();
        let mut output = ndarray::Array2::<f64>::zeros((n_samples, total_categories));

        output.axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .try_for_each(|(i, mut row)| -> Result<(), String> {
                let mut offset = 0;
                for j in 0..n_features {
                    let val = X[[i, j]];
                    match &self.categories[j] {
                        ColumnCategories::Int(cats) => {
                            match cats.binary_search(&val) {
                                Ok(idx) => {
                                    row[offset + idx] = 1.0;
                                }
                                Err(_) => {
                                    if self.handle_unknown == "error" {
                                        return Err(format!("Found unknown category {} in column {}", val, j));
                                    }
                                }
                            }
                            offset += cats.len();
                        }
                        _ => return Err("Expected integer category, found string categories".to_string()),
                    }
                }
                Ok(())
            })?;

        Ok(output)
    }

    pub fn transform_str(&self, X_cols: &[Vec<String>]) -> Result<ndarray::Array2<f64>, String> {
        if X_cols.is_empty() {
            return Err("Input is empty".to_string());
        }
        let n_features = X_cols.len();
        let n_samples = X_cols[0].len();

        if n_features != self.categories.len() {
            return Err(format!("Feature mismatch: expected {}, got {}", self.categories.len(), n_features));
        }

        let total_categories: usize = self.categories.iter().map(|c| c.len()).sum();
        let mut output = ndarray::Array2::<f64>::zeros((n_samples, total_categories));

        output.axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .try_for_each(|(i, mut row)| -> Result<(), String> {
                let mut offset = 0;
                for j in 0..n_features {
                    let val = &X_cols[j][i];
                    match &self.categories[j] {
                        ColumnCategories::Str(cats) => {
                            match cats.binary_search(val) {
                                Ok(idx) => {
                                    row[offset + idx] = 1.0;
                                }
                                Err(_) => {
                                    if self.handle_unknown == "error" {
                                        return Err(format!("Found unknown category {} in column {}", val, j));
                                    }
                                }
                            }
                            offset += cats.len();
                        }
                        _ => return Err("Expected string category, found integer categories".to_string()),
                    }
                }
                Ok(())
            })?;

        Ok(output)
    }

    pub fn inverse_transform_int(&self, X: &ArrayView2<f64>) -> Result<ndarray::Array2<i64>, String> {
        let n_samples = X.nrows();
        let n_features = self.categories.len();
        let mut output = ndarray::Array2::<i64>::zeros((n_samples, n_features));

        output.axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .try_for_each(|(i, mut row)| -> Result<(), String> {
                let mut offset = 0;
                for j in 0..n_features {
                    match &self.categories[j] {
                        ColumnCategories::Int(cats) => {
                            let sub_slice = X.slice(ndarray::s![i, offset..offset + cats.len()]);
                            let mut max_idx = 0;
                            let mut max_val = sub_slice[0];
                            for (k, &val) in sub_slice.iter().enumerate() {
                                if val > max_val {
                                    max_val = val;
                                    max_idx = k;
                                }
                            }
                            row[j] = cats[max_idx];
                            offset += cats.len();
                        }
                        _ => return Err("Expected integer categories".to_string()),
                    }
                }
                Ok(())
            })?;

        Ok(output)
    }

    pub fn inverse_transform_str(&self, X: &ArrayView2<f64>) -> Result<Vec<Vec<String>>, String> {
        let n_samples = X.nrows();
        let n_features = self.categories.len();
        let mut output = vec![vec![String::new(); n_features]; n_samples];

        output.par_iter_mut()
            .enumerate()
            .try_for_each(|(i, row)| -> Result<(), String> {
                let mut offset = 0;
                for j in 0..n_features {
                    match &self.categories[j] {
                        ColumnCategories::Str(cats) => {
                            let sub_slice = X.slice(ndarray::s![i, offset..offset + cats.len()]);
                            let mut max_idx = 0;
                            let mut max_val = sub_slice[0];
                            for (k, &val) in sub_slice.iter().enumerate() {
                                if val > max_val {
                                    max_val = val;
                                    max_idx = k;
                                }
                            }
                            row[j] = cats[max_idx].clone();
                            offset += cats.len();
                        }
                        _ => return Err("Expected string categories".to_string()),
                    }
                }
                Ok(())
            })?;

        Ok(output)
    }
}
```

---

### B. PyO3 Binding Layer (`crates/thermite-binding`)

#### File: `crates/thermite-binding/src/lib.rs`
```rust
use pyo3::prelude::*;
use pyo3::types::{PyTuple, PyList};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};

use thermite_core::model_selection::compute_split_indices;
use thermite_core::preprocessing::StandardScaler as CoreStandardScaler;
use thermite_core::preprocessing::MinMaxScaler as CoreMinMaxScaler;
use thermite_core::preprocessing::LabelEncoderCore;
use thermite_core::preprocessing::OneHotEncoderCore;

#[pyfunction]
fn ping() -> PyResult<String> {
    Ok("pong".to_string())
}

#[pyfunction]
#[pyo3(signature = (*arrays, test_size=None, train_size=None, random_state=None, shuffle=true, stratify=None))]
fn train_test_split<'py>(
    py: Python<'py>,
    arrays: &Bound<'py, PyTuple>,
    test_size: Option<f64>,
    train_size: Option<f64>,
    random_state: Option<u64>,
    shuffle: bool,
    stratify: Option<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyTuple>> {
    if arrays.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("At least one array is required as input"));
    }

    let first_arr = arrays.get_item(0)?;
    let len_any: usize = first_arr.len()?;
    
    for i in 1..arrays.len() {
        let arr = arrays.get_item(i)?;
        if arr.len()? != len_any {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "All input arrays must have the same length"
            ));
        }
    }

    let stratify_vec = if let Some(strat_obj) = stratify {
        let mut labels = Vec::new();
        let len = strat_obj.len()?;
        if len != len_any {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Stratify labels length must match number of samples"
            ));
        }
        for i in 0..len {
            let item = strat_obj.get_item(i)?;
            let item_str = item.str()?.to_string();
            labels.push(item_str);
        }
        Some(labels)
    } else {
        None
    };

    let split = compute_split_indices(
        len_any,
        test_size,
        train_size,
        shuffle,
        random_state,
        stratify_vec.as_deref(),
    ).map_err(pyo3::exceptions::PyValueError::new_err)?;

    let train_idx_py = PyArray1::from_vec(py, split.train_indices);
    let test_idx_py = PyArray1::from_vec(py, split.test_indices);

    let mut results = Vec::with_capacity(arrays.len() * 2);
    for i in 0..arrays.len() {
        let arr = arrays.get_item(i)?;
        let train_slice = arr.call_method1("__getitem__", (train_idx_py.clone(),))?;
        let test_slice = arr.call_method1("__getitem__", (test_idx_py.clone(),))?;
        results.push(train_slice);
        results.push(test_slice);
    }

    Ok(PyTuple::new(py, results))
}

#[pyclass]
pub struct StandardScaler {
    core: CoreStandardScaler,
}

#[pymethods]
impl StandardScaler {
    #[new]
    #[pyo3(signature = (with_mean=true, with_std=true))]
    fn new(with_mean: bool, with_std: bool) -> Self {
        StandardScaler {
            core: CoreStandardScaler::new(with_mean, with_std),
        }
    }

    fn fit(&mut self, X: PyReadonlyArray2<f64>) -> PyResult<()> {
        let x_arr = X.as_array();
        self.core.fit(&x_arr).map_err(pyo3::exceptions::PyValueError::new_err)
    }

    fn transform<'py>(&self, py: Python<'py>, X: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x_arr = X.as_array();
        let out = self.core.transform(&x_arr).map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyArray2::from_array(py, &out))
    }

    fn fit_transform<'py>(&mut self, py: Python<'py>, X: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x_arr = X.as_array();
        self.core.fit(&x_arr).map_err(pyo3::exceptions::PyValueError::new_err)?;
        let out = self.core.transform(&x_arr).map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyArray2::from_array(py, &out))
    }

    fn inverse_transform<'py>(&self, py: Python<'py>, X: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x_arr = X.as_array();
        let out = self.core.inverse_transform(&x_arr).map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyArray2::from_array(py, &out))
    }

    #[getter]
    fn mean<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        match &self.core.mean {
            Some(m) => Ok(Some(PyArray1::from_array(py, m))),
            None => Ok(None),
        }
    }

    #[getter]
    fn var<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        match &self.core.var {
            Some(v) => Ok(Some(PyArray1::from_array(py, v))),
            None => Ok(None),
        }
    }

    #[getter]
    fn scale<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        match &self.core.scale {
            Some(s) => Ok(Some(PyArray1::from_array(py, s))),
            None => Ok(None),
        }
    }

    #[getter]
    fn n_samples_seen(&self) -> usize {
        self.core.n_samples_seen
    }
}

#[pyclass]
pub struct MinMaxScaler {
    core: CoreMinMaxScaler,
}

#[pymethods]
impl MinMaxScaler {
    #[new]
    #[pyo3(signature = (feature_range=(0.0, 1.0)))]
    fn new(feature_range: (f64, f64)) -> PyResult<Self> {
        let core = CoreMinMaxScaler::new(feature_range).map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(MinMaxScaler { core })
    }

    fn fit(&mut self, X: PyReadonlyArray2<f64>) -> PyResult<()> {
        let x_arr = X.as_array();
        self.core.fit(&x_arr).map_err(pyo3::exceptions::PyValueError::new_err)
    }

    fn transform<'py>(&self, py: Python<'py>, X: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x_arr = X.as_array();
        let out = self.core.transform(&x_arr).map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyArray2::from_array(py, &out))
    }

    fn fit_transform<'py>(&mut self, py: Python<'py>, X: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x_arr = X.as_array();
        self.core.fit(&x_arr).map_err(pyo3::exceptions::PyValueError::new_err)?;
        let out = self.core.transform(&x_arr).map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyArray2::from_array(py, &out))
    }

    fn inverse_transform<'py>(&self, py: Python<'py>, X: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x_arr = X.as_array();
        let out = self.core.inverse_transform(&x_arr).map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyArray2::from_array(py, &out))
    }

    #[getter]
    fn data_min<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        match &self.core.data_min {
            Some(dm) => Ok(Some(PyArray1::from_array(py, dm))),
            None => Ok(None),
        }
    }

    #[getter]
    fn data_max<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        match &self.core.data_max {
            Some(dm) => Ok(Some(PyArray1::from_array(py, dm))),
            None => Ok(None),
        }
    }

    #[getter]
    fn scale<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        match &self.core.scale {
            Some(s) => Ok(Some(PyArray1::from_array(py, s))),
            None => Ok(None),
        }
    }

    #[getter]
    fn min<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        match &self.core.min {
            Some(m) => Ok(Some(PyArray1::from_array(py, m))),
            None => Ok(None),
        }
    }
}

#[pyclass]
pub struct LabelEncoder {
    core: LabelEncoderCore,
}

#[pymethods]
impl LabelEncoder {
    #[new]
    fn new() -> Self {
        LabelEncoder {
            core: LabelEncoderCore::new(),
        }
    }

    fn fit_int(&mut self, y: PyReadonlyArray1<i64>) {
        let y_view = y.as_array();
        let y_slice = y_view.as_slice().unwrap_or(&y_view.to_vec());
        self.core.fit_int(y_slice);
    }

    fn fit_str(&mut self, y: Vec<String>) {
        self.core.fit_str(y);
    }

    fn transform_int<'py>(&self, py: Python<'py>, y: PyReadonlyArray1<i64>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let y_view = y.as_array();
        let y_slice = y_view.as_slice().unwrap_or(&y_view.to_vec());
        let out = self.core.transform_int(y_slice).map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyArray1::from_vec(py, out))
    }

    fn transform_str<'py>(&self, py: Python<'py>, y: Vec<String>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let out = self.core.transform_str(&y).map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyArray1::from_vec(py, out))
    }

    fn inverse_transform_int<'py>(&self, py: Python<'py>, y: PyReadonlyArray1<i64>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let y_view = y.as_array();
        let y_slice = y_view.as_slice().unwrap_or(&y_view.to_vec());
        let out = self.core.inverse_transform_int(y_slice).map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyArray1::from_vec(py, out))
    }

    fn inverse_transform_str(&self, y: PyReadonlyArray1<i64>) -> PyResult<Vec<String>> {
        let y_view = y.as_array();
        let y_slice = y_view.as_slice().unwrap_or(&y_view.to_vec());
        self.core.inverse_transform_str(y_slice).map_err(pyo3::exceptions::PyValueError::new_err)
    }

    fn get_classes_int(&self) -> Option<Vec<i64>> {
        self.core.classes_int.clone()
    }

    fn get_classes_str(&self) -> Option<Vec<String>> {
        self.core.classes_str.clone()
    }
}

#[pyclass]
pub struct OneHotEncoder {
    core: OneHotEncoderCore,
}

#[pymethods]
impl OneHotEncoder {
    #[new]
    #[pyo3(signature = (handle_unknown="error"))]
    fn new(handle_unknown: &str) -> PyResult<Self> {
        if handle_unknown != "error" && handle_unknown != "ignore" {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "handle_unknown must be either 'error' or 'ignore'"
            ));
        }
        Ok(OneHotEncoder {
            core: OneHotEncoderCore::new(handle_unknown.to_string()),
        })
    }

    fn fit_int(&mut self, X: PyReadonlyArray2<i64>) {
        let x_arr = X.as_array();
        self.core.fit_int(&x_arr);
    }

    fn fit_str(&mut self, X_cols: Vec<Vec<String>>) {
        self.core.fit_str(X_cols);
    }

    fn transform_int<'py>(&self, py: Python<'py>, X: PyReadonlyArray2<i64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x_arr = X.as_array();
        let out = self.core.transform_int(&x_arr).map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyArray2::from_array(py, &out))
    }

    fn transform_str<'py>(&self, py: Python<'py>, X_cols: Vec<Vec<String>>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let out = self.core.transform_str(&X_cols).map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyArray2::from_array(py, &out))
    }

    fn inverse_transform_int<'py>(&self, py: Python<'py>, X: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray2<i64>>> {
        let x_arr = X.as_array();
        let out = self.core.inverse_transform_int(&x_arr).map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyArray2::from_array(py, &out))
    }

    fn inverse_transform_str(&self, X: PyReadonlyArray2<f64>) -> PyResult<Vec<Vec<String>>> {
        let x_arr = X.as_array();
        self.core.inverse_transform_str(&x_arr).map_err(pyo3::exceptions::PyValueError::new_err)
    }

    #[getter]
    fn categories<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::empty(py);
        for cat in &self.core.categories {
            match cat {
                thermite_core::preprocessing::ColumnCategories::Int(cats) => {
                    list.append(PyArray1::from_vec(py, cats.clone()))?;
                }
                thermite_core::preprocessing::ColumnCategories::Str(cats) => {
                    list.append(cats.clone())?;
                }
            }
        }
        Ok(list)
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ping, m)?)?;
    m.add_function(wrap_pyfunction!(train_test_split, m)?)?;
    m.add_class::<StandardScaler>()?;
    m.add_class::<MinMaxScaler>()?;
    m.add_class::<LabelEncoder>()?;
    m.add_class::<OneHotEncoder>()?;
    Ok(())
}
```

---

### C. Python Packages API Layer (`thermite/`)

#### File: `thermite/model_selection.py`
```python
import numpy as np
from . import _core

def train_test_split(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None):
    """Split arrays or matrices into random train and test subsets.
    
    Compatible with scikit-learn.
    """
    if len(arrays) == 0:
        raise ValueError("At least one array is required as input")
        
    np_arrays = [np.asarray(arr) for arr in arrays]
    
    if stratify is not None:
        stratify = np.asarray(stratify)
        
    return _core.train_test_split(
        *np_arrays,
        test_size=test_size,
        train_size=train_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=stratify
    )
```

#### File: `thermite/preprocessing.py`
```python
import numpy as np
from . import _core

class StandardScaler:
    def __init__(self, *, with_mean=True, with_std=True):
        self._scaler = _core.StandardScaler(with_mean=with_mean, with_std=with_std)
        
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        self._scaler.fit(X)
        return self
        
    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        return self._scaler.transform(X)
        
    def fit_transform(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        return self._scaler.fit_transform(X)
        
    def inverse_transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        return self._scaler.inverse_transform(X)
        
    @property
    def mean_(self):
        return self._scaler.mean
        
    @property
    def var_(self):
        return self._scaler.var
        
    @property
    def scale_(self):
        return self._scaler.scale
        
    @property
    def n_samples_seen_(self):
        return self._scaler.n_samples_seen

class MinMaxScaler:
    def __init__(self, feature_range=(0.0, 1.0)):
        self._scaler = _core.MinMaxScaler(feature_range=feature_range)
        
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        self._scaler.fit(X)
        return self
        
    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        return self._scaler.transform(X)
        
    def fit_transform(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        return self._scaler.fit_transform(X)
        
    def inverse_transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
        return self._scaler.inverse_transform(X)
        
    @property
    def data_min_(self):
        return self._scaler.data_min
        
    @property
    def data_max_(self):
        return self._scaler.data_max
        
    @property
    def scale_(self):
        return self._scaler.scale
        
    @property
    def min_(self):
        return self._scaler.min

class LabelEncoder:
    def __init__(self):
        self._encoder = _core.LabelEncoder()
        self.classes_ = None
        
    def fit(self, y):
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError("Expected 1D array for y")
            
        if np.issubdtype(y.dtype, np.integer):
            y_cast = y.astype(np.int64)
            self._encoder.fit_int(y_cast)
            self.classes_ = np.array(self._encoder.get_classes_int())
        else:
            y_cast = list(y.astype(str))
            self._encoder.fit_str(y_cast)
            self.classes_ = np.array(self._encoder.get_classes_str())
        return self
        
    def transform(self, y):
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError("Expected 1D array for y")
            
        if np.issubdtype(y.dtype, np.integer):
            y_cast = y.astype(np.int64)
            return self._encoder.transform_int(y_cast)
        else:
            y_cast = list(y.astype(str))
            return self._encoder.transform_str(y_cast)
            
    def fit_transform(self, y):
        return self.fit(y).transform(y)
        
    def inverse_transform(self, y):
        y = np.asarray(y, dtype=np.int64)
        if y.ndim != 1:
            raise ValueError("Expected 1D array for y")
            
        if self.classes_.dtype.kind in ('i', 'u'):
            return self._encoder.inverse_transform_int(y)
        else:
            return np.array(self._encoder.inverse_transform_str(y))

class OneHotEncoder:
    def __init__(self, *, handle_unknown="error"):
        if handle_unknown not in ("error", "ignore"):
            raise ValueError("handle_unknown must be 'error' or 'ignore'")
        self._encoder = _core.OneHotEncoder(handle_unknown=handle_unknown)
        self.categories_ = None
        
    def fit(self, X, y=None):
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
            
        if np.issubdtype(X.dtype, np.integer):
            self._encoder.fit_int(X.astype(np.int64))
        else:
            X_str = X.astype(str)
            cols = [list(X_str[:, i]) for i in range(X_str.shape[1])]
            self._encoder.fit_str(cols)
            
        raw_cats = self._encoder.categories
        self.categories_ = [np.array(c) for c in raw_cats]
        return self
        
    def transform(self, X):
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
            
        if np.issubdtype(X.dtype, np.integer):
            return self._encoder.transform_int(X.astype(np.int64))
        else:
            X_str = X.astype(str)
            cols = [list(X_str[:, i]) for i in range(X_str.shape[1])]
            return self._encoder.transform_str(cols)
            
    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
        
    def inverse_transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("Expected 2D array for X")
            
        if self.categories_ is None or len(self.categories_) == 0:
            raise ValueError("OneHotEncoder is not fitted yet")
            
        is_int = np.issubdtype(self.categories_[0].dtype, np.integer)
        if is_int:
            return self._encoder.inverse_transform_int(X)
        else:
            return np.array(self._encoder.inverse_transform_str(X))
```

#### File: `thermite/__init__.py`
```python
from ._core import ping
from .preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from .model_selection import train_test_split

__version__ = "0.1.0"
__all__ = [
    "ping",
    "StandardScaler",
    "MinMaxScaler",
    "LabelEncoder",
    "OneHotEncoder",
    "train_test_split"
]
```

---

## 5. Verification Method

### A. Independent Rust Unit-Tests
Unit tests should be placed in the respective Rust source files. They can be executed using standard `cargo test` command in root.

Example test suite to add in `crates/thermite-core/src/model_selection.rs`:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_indices_basic() {
        let indices = compute_split_indices(100, Some(0.20), None, true, Some(42), None).unwrap();
        assert_eq!(indices.train_indices.len(), 80);
        assert_eq!(indices.test_indices.len(), 20);
        
        // Ensure no overlap
        for idx in &indices.train_indices {
            assert!(!indices.test_indices.contains(idx));
        }
    }

    #[test]
    fn test_split_indices_stratified() {
        let labels: Vec<String> = (0..100).map(|i| if i % 2 == 0 { "A" } else { "B" }.to_string()).collect();
        let indices = compute_split_indices(100, Some(0.50), None, true, Some(42), Some(&labels)).unwrap();
        
        assert_eq!(indices.train_indices.len(), 50);
        assert_eq!(indices.test_indices.len(), 50);
        
        let train_a = indices.train_indices.iter().filter(|&&i| labels[i] == "A").count();
        let test_a = indices.test_indices.iter().filter(|&&i| labels[i] == "A").count();
        assert_eq!(train_a, 25);
        assert_eq!(test_a, 25);
    }
}
```

### B. Python Compatibility Test Suite
Run a verification script that compares thermite outputs with scikit-learn outputs directly:

```python
import numpy as np
import sklearn.preprocessing as sk_pre
import sklearn.model_selection as sk_mod
import thermite.preprocessing as th_pre
import thermite.model_selection as th_mod

# 1. train_test_split test
X = np.random.randn(100, 4)
y = np.random.choice([0, 1], size=100)
X_tr, X_te, y_tr, y_te = th_mod.train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
assert X_tr.shape == (80, 4)
assert X_te.shape == (20, 4)
# Check stratify ratios
assert np.sum(y_tr == 1) / len(y_tr) == np.sum(y_te == 1) / len(y_te)

# 2. StandardScaler test
X_scale = np.random.randn(200, 5)
sk_scaler = sk_pre.StandardScaler().fit(X_scale)
th_scaler = th_pre.StandardScaler().fit(X_scale)
assert np.allclose(sk_scaler.mean_, th_scaler.mean_)
assert np.allclose(sk_scaler.var_, th_scaler.var_)
assert np.allclose(sk_scaler.transform(X_scale), th_scaler.transform(X_scale))
assert np.allclose(X_scale, th_scaler.inverse_transform(th_scaler.transform(X_scale)))

# 3. MinMaxScaler test
sk_minmax = sk_pre.MinMaxScaler(feature_range=(-1, 1)).fit(X_scale)
th_minmax = th_pre.MinMaxScaler(feature_range=(-1, 1)).fit(X_scale)
assert np.allclose(sk_minmax.data_min_, th_minmax.data_min_)
assert np.allclose(sk_minmax.data_max_, th_minmax.data_max_)
assert np.allclose(sk_minmax.transform(X_scale), th_minmax.transform(X_scale))

# 4. LabelEncoder test
y_labels = np.array(["cat", "dog", "cat", "bird", "dog"])
sk_le = sk_pre.LabelEncoder().fit(y_labels)
th_le = th_pre.LabelEncoder().fit(y_labels)
assert np.array_equal(sk_le.classes_, th_le.classes_)
assert np.array_equal(sk_le.transform(y_labels), th_le.transform(y_labels))

# 5. OneHotEncoder test
X_cat = np.array([["int_1", "str_A"], ["int_2", "str_B"], ["int_1", "str_A"]], dtype=object)
sk_ohe = sk_pre.OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(X_cat)
th_ohe = th_pre.OneHotEncoder(handle_unknown='ignore').fit(X_cat)
assert np.allclose(sk_ohe.transform(X_cat), th_ohe.transform(X_cat))
print("All Python verification tests pass!")
```
