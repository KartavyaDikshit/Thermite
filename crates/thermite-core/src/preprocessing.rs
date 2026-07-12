#![allow(non_snake_case)]
use ndarray::{Array1, ArrayView2, Axis};
use rayon::prelude::*;


// Helper to check for non-finite numbers (NaN/Inf)
fn check_finite(X: &ArrayView2<f64>) -> Result<(), String> {
    if X.iter().any(|&val| !val.is_finite()) {
        return Err("Input contains NaN or infinity values".to_string());
    }
    Ok(())
}

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
        check_finite(X)?;

        let n_samples = X.nrows();
        let n_features = X.ncols();

        // Parallel column-wise statistics
        let stats: Vec<(f64, f64)> = X.axis_iter(Axis(1))
            .into_par_iter()
            .map(|col| {
                let n = col.len() as f64;
                if n == 0.0 {
                    return (0.0, 0.0);
                }
                let mean = col.sum() / n;
                let var = col.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
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
        check_finite(X)?;

        let n_features = X.ncols();
        let mean = self.mean.as_ref().unwrap();
        let scale = self.scale.as_ref().unwrap();

        if n_features != mean.len() {
            return Err(format!("Feature mismatch: expected {}, got {}", mean.len(), n_features));
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
        check_finite(X)?;

        let n_features = X.ncols();
        let mean = self.mean.as_ref().unwrap();
        let scale = self.scale.as_ref().unwrap();

        if n_features != mean.len() {
            return Err(format!("Feature mismatch: expected {}, got {}", mean.len(), n_features));
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
        check_finite(X)?;

        let n_features = X.ncols();

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
                scale_arr[i] = 1.0;
                min_arr[i] = min_val - d_min * 1.0;
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
        check_finite(X)?;

        let n_features = X.ncols();
        let scale = self.scale.as_ref().unwrap();
        let min = self.min.as_ref().unwrap();

        if n_features != scale.len() {
            return Err(format!("Feature mismatch: expected {}, got {}", scale.len(), n_features));
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
        check_finite(X)?;

        let n_features = X.ncols();
        let scale = self.scale.as_ref().unwrap();
        let min = self.min.as_ref().unwrap();
        let data_min = self.data_min.as_ref().unwrap();

        if n_features != scale.len() {
            return Err(format!("Feature mismatch: expected {}, got {}", scale.len(), n_features));
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
    pub classes_float: Option<Vec<f64>>,
}

impl LabelEncoderCore {
    pub fn new() -> Self {
        LabelEncoderCore {
            classes_int: None,
            classes_str: None,
            classes_float: None,
        }
    }

    pub fn fit_int(&mut self, y: &[i64]) {
        let mut sorted = y.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        self.classes_int = Some(sorted);
        self.classes_str = None;
        self.classes_float = None;
    }

    pub fn fit_str(&mut self, y: Vec<String>) {
        let mut sorted = y;
        sorted.sort_unstable();
        sorted.dedup();
        self.classes_str = Some(sorted);
        self.classes_int = None;
        self.classes_float = None;
    }

    pub fn fit_float(&mut self, y: &[f64]) {
        let mut sorted = y.to_vec();
        sorted.sort_unstable_by(|a, b| a.total_cmp(b));
        sorted.dedup();
        self.classes_float = Some(sorted);
        self.classes_int = None;
        self.classes_str = None;
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

    pub fn transform_float(&self, y: &[f64]) -> Result<Vec<i64>, String> {
        let classes = self.classes_float.as_ref().ok_or("LabelEncoder not fitted on floats")?;
        let mut out = Vec::with_capacity(y.len());
        for &val in y {
            match classes.binary_search_by(|probe| probe.total_cmp(&val)) {
                Ok(idx) => out.push(idx as i64),
                Err(_) => return Err(format!("y contains previously unseen labels: {}", val)),
            }
        }
        Ok(out)
    }

    pub fn inverse_transform_float(&self, y: &[i64]) -> Result<Vec<f64>, String> {
        let classes = self.classes_float.as_ref().ok_or("LabelEncoder not fitted on floats")?;
        let mut out = Vec::with_capacity(y.len());
        for &idx in y {
            if idx < 0 || idx >= classes.len() as i64 {
                return Err(format!("y contains out-of-bounds index: {}", idx));
            }
            out.push(classes[idx as usize]);
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
        
        let total_categories: usize = self.categories.iter().map(|c| c.len()).sum();
        if X.ncols() != total_categories {
            return Err(format!("Feature mismatch: expected {} one-hot encoded columns, got {}", total_categories, X.ncols()));
        }

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

        let total_categories: usize = self.categories.iter().map(|c| c.len()).sum();
        if X.ncols() != total_categories {
            return Err(format!("Feature mismatch: expected {} one-hot encoded columns, got {}", total_categories, X.ncols()));
        }

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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_standard_scaler() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let mut scaler = StandardScaler::new(true, true);
        scaler.fit(&x.view()).unwrap();

        assert_eq!(scaler.mean.as_ref().unwrap(), &array![3.0, 4.0]);
        // Var is [(1-3)^2 + 0 + (5-3)^2]/3 = 8/3 = 2.6666666666666665
        assert!((scaler.var.as_ref().unwrap()[0] - 8.0/3.0).abs() < 1e-9);

        let xt = scaler.transform(&x.view()).unwrap();
        // first col mean 3, std sqrt(8/3)
        // [1.0, 3.0, 5.0] -> [-1.22474487, 0.0, 1.22474487]
        assert!((xt[[0, 0]] + 1.224744871391589).abs() < 1e-7);

        let x_orig = scaler.inverse_transform(&xt.view()).unwrap();
        assert!((x_orig[[0, 0]] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_min_max_scaler() {
        let x = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]];
        let mut scaler = MinMaxScaler::new((0.0, 1.0)).unwrap();
        scaler.fit(&x.view()).unwrap();

        assert_eq!(scaler.data_min.as_ref().unwrap(), &array![1.0, 10.0]);
        assert_eq!(scaler.data_max.as_ref().unwrap(), &array![3.0, 30.0]);

        let xt = scaler.transform(&x.view()).unwrap();
        assert_eq!(xt, array![[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]);

        let x_orig = scaler.inverse_transform(&xt.view()).unwrap();
        assert_eq!(x_orig, x);
    }

    #[test]
    fn test_label_encoder() {
        let mut le = LabelEncoderCore::new();
        le.fit_int(&[20, 10, 20, 30]);
        assert_eq!(le.classes_int.as_ref().unwrap(), &vec![10, 20, 30]);

        let encoded = le.transform_int(&[30, 10, 20]).unwrap();
        assert_eq!(encoded, vec![2, 0, 1]);

        let decoded = le.inverse_transform_int(&[2, 0, 1]).unwrap();
        assert_eq!(decoded, vec![30, 10, 20]);
    }

    #[test]
    fn test_one_hot_encoder() {
        let mut ohe = OneHotEncoderCore::new("error".to_string());
        let x = array![[1, 10], [2, 20], [1, 20]];
        ohe.fit_int(&x.view());
        // col 0 cats: 1, 2 (len 2)
        // col 1 cats: 10, 20 (len 2)
        let xt = ohe.transform_int(&x.view()).unwrap();
        // first row [1, 10] -> [1, 0, 1, 0]
        assert_eq!(xt, array![[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]]);

        let x_orig = ohe.inverse_transform_int(&xt.view()).unwrap();
        assert_eq!(x_orig, x);
    }
}
