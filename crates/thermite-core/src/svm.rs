#![allow(non_snake_case)]
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use std::os::raw::c_int;

#[repr(C)]
#[derive(Debug)]
pub struct SvmModel {
    pub kernel_type: c_int,
    pub degree: c_int,
    pub gamma: f64,
    pub coef0: f64,

    pub l: c_int,
    pub D: c_int,
    pub SVs: *mut f64,
    pub sv_coef: *mut f64,
    pub rho: f64,

    pub probA: f64,
    pub probB: f64,
}

extern "C" {
    pub fn svm_train(
        X: *const f64,
        y: *const f64,
        N: c_int,
        D: c_int,
        C: f64,
        kernel_type: c_int,
        degree: c_int,
        gamma: f64,
        coef0: f64,
        eps: f64,
        max_iter: c_int,
        probability: c_int,
    ) -> *mut SvmModel;

    pub fn svm_predict_decision(model: *const SvmModel, x: *const f64) -> f64;
    pub fn svm_free_model(model: *mut SvmModel);
}

#[derive(Debug)]
pub struct BinarySVC {
    pub model_ptr: *mut SvmModel,
    pub class_i: f64,
    pub class_j: f64,
}

impl Drop for BinarySVC {
    fn drop(&mut self) {
        unsafe {
            if !self.model_ptr.is_null() {
                svm_free_model(self.model_ptr);
            }
        }
    }
}

unsafe impl Send for BinarySVC {}
unsafe impl Sync for BinarySVC {}

#[derive(Debug)]
pub struct SVC {
    pub C: f64,
    pub kernel: String,
    pub degree: i32,
    pub gamma: String, // "scale", "auto", or a number as a string
    pub coef0: f64,
    pub probability: bool,
    pub eps: f64,
    pub max_iter: i32,

    pub classes_: Option<Vec<f64>>,
    pub models_: Option<Vec<BinarySVC>>,
    pub n_features_: Option<usize>,
}

impl SVC {
    pub fn new(
        C: f64,
        kernel: String,
        degree: i32,
        gamma: String,
        coef0: f64,
        probability: bool,
        eps: f64,
        max_iter: i32,
    ) -> Self {
        SVC {
            C,
            kernel,
            degree,
            gamma,
            coef0,
            probability,
            eps,
            max_iter,
            classes_: None,
            models_: None,
            n_features_: None,
        }
    }

    pub fn fit(&mut self, X: &ArrayView2<f64>, y: &ArrayView1<f64>) -> Result<(), String> {
        if X.is_empty() {
            return Err("Input features X is empty".to_string());
        }
        if y.is_empty() {
            return Err("Input targets y is empty".to_string());
        }
        if X.nrows() != y.len() {
            return Err("Length mismatch between X and y".to_string());
        }

        // Get unique sorted classes
        let mut unique_classes = y.to_vec();
        unique_classes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        unique_classes.dedup_by(|a, b| (*a - *b).abs() < 1e-9);

        let n_classes = unique_classes.len();
        if n_classes < 2 {
            return Err("SVC requires at least 2 classes".to_string());
        }

        // Compute gamma value
        let n_features = X.ncols() as f64;
        let gamma_val = match self.gamma.as_str() {
            "auto" => 1.0 / n_features,
            "scale" => {
                let mean = X.mean().unwrap_or(0.0);
                let var = X.fold(0.0, |acc, &val| acc + (val - mean).powi(2)) / (X.len() as f64);
                if var > 0.0 {
                    1.0 / (n_features * var)
                } else {
                    1.0 / n_features
                }
            }
            other => other
                .parse::<f64>()
                .map_err(|e| format!("Invalid gamma value: {}", e))?,
        };

        let kernel_type = match self.kernel.as_str() {
            "rbf" => 0,
            "poly" => 1,
            _ => return Err(format!("Unsupported kernel type: {}", self.kernel)),
        };

        let prob_flag = if self.probability { 1 } else { 0 };
        let mut models = Vec::new();

        // One-vs-One (OvO) binary classifiers
        for i in 0..n_classes {
            for j in (i + 1)..n_classes {
                let class_i = unique_classes[i];
                let class_j = unique_classes[j];

                // Filter data for class_i and class_j
                let mut sub_X = Vec::new();
                let mut sub_y = Vec::new();
                for row_idx in 0..X.nrows() {
                    let label = y[row_idx];
                    let is_i = (label - class_i).abs() < 1e-9;
                    let is_j = (label - class_j).abs() < 1e-9;
                    if is_i || is_j {
                        for &val in X.row(row_idx) {
                            sub_X.push(val);
                        }
                        sub_y.push(if is_i { 1.0 } else { -1.0 });
                    }
                }

                let n_samples_sub = sub_y.len() as c_int;
                let model_ptr = unsafe {
                    svm_train(
                        sub_X.as_ptr(),
                        sub_y.as_ptr(),
                        n_samples_sub,
                        X.ncols() as c_int,
                        self.C,
                        kernel_type,
                        self.degree as c_int,
                        gamma_val,
                        self.coef0,
                        self.eps,
                        self.max_iter as c_int,
                        prob_flag,
                    )
                };

                if model_ptr.is_null() {
                    return Err("Failed to train SVM binary model".to_string());
                }

                models.push(BinarySVC {
                    model_ptr,
                    class_i,
                    class_j,
                });
            }
        }

        self.classes_ = Some(unique_classes);
        self.models_ = Some(models);
        self.n_features_ = Some(X.ncols());

        Ok(())
    }

    pub fn predict(&self, X: &ArrayView2<f64>) -> Result<Array1<f64>, String> {
        let classes = self.classes_.as_ref().ok_or("Model is not fitted yet")?;
        let models = self.models_.as_ref().ok_or("Model is not fitted yet")?;
        if let Some(n_feat) = self.n_features_ {
            if X.ncols() != n_feat {
                return Err(format!("X has {} features, but SVC is expecting {} features as input", X.ncols(), n_feat));
            }
        }

        let n_samples = X.nrows();
        let mut predictions = Array1::<f64>::zeros(n_samples);

        let K = classes.len();

        for i in 0..n_samples {
            let row = X.row(i);
            let mut row_vec = Vec::with_capacity(row.len());
            for &val in row {
                row_vec.push(val);
            }

            let mut votes = vec![0; K];
            for model in models {
                let dec = unsafe { svm_predict_decision(model.model_ptr, row_vec.as_ptr()) };
                let is_i = dec >= 0.0;
                let class_to_vote = if is_i { model.class_i } else { model.class_j };

                if let Some(idx) = classes
                    .iter()
                    .position(|&c| (c - class_to_vote).abs() < 1e-9)
                {
                    votes[idx] += 1;
                }
            }

            let max_idx = votes
                .iter()
                .enumerate()
                .max_by_key(|&(_, &v)| v)
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            predictions[i] = classes[max_idx];
        }

        Ok(predictions)
    }

    pub fn predict_proba(&self, X: &ArrayView2<f64>) -> Result<Array2<f64>, String> {
        if !self.probability {
            return Err("predict_proba is not available when probability=false".to_string());
        }
        let classes = self.classes_.as_ref().ok_or("Model is not fitted yet")?;
        let models = self.models_.as_ref().ok_or("Model is not fitted yet")?;
        if let Some(n_feat) = self.n_features_ {
            if X.ncols() != n_feat {
                return Err(format!("X has {} features, but SVC is expecting {} features as input", X.ncols(), n_feat));
            }
        }

        let n_samples = X.nrows();
        let K = classes.len();
        let mut proba = Array2::<f64>::zeros((n_samples, K));

        for i in 0..n_samples {
            let row = X.row(i);
            let mut row_vec = Vec::with_capacity(row.len());
            for &val in row {
                row_vec.push(val);
            }

            let mut r = Array2::<f64>::zeros((K, K));
            for model in models {
                let dec = unsafe { svm_predict_decision(model.model_ptr, row_vec.as_ptr()) };
                let probA = unsafe { (*model.model_ptr).probA };
                let probB = unsafe { (*model.model_ptr).probB };

                let arg = probA * dec + probB;
                let p = 1.0 / (1.0 + f64::exp(arg.max(-50.0).min(50.0)));

                let idx_i = classes
                    .iter()
                    .position(|&c| (c - model.class_i).abs() < 1e-9)
                    .unwrap();
                let idx_j = classes
                    .iter()
                    .position(|&c| (c - model.class_j).abs() < 1e-9)
                    .unwrap();

                r[[idx_i, idx_j]] = p;
                r[[idx_j, idx_i]] = 1.0 - p;
            }

            let mut s = vec![0.0; K];
            for idx_i in 0..K {
                for idx_j in 0..K {
                    if idx_i != idx_j {
                        s[idx_i] += r[[idx_i, idx_j]];
                    }
                }
            }

            let sum_s: f64 = s.iter().sum();
            if sum_s > 0.0 {
                for idx_i in 0..K {
                    proba[[i, idx_i]] = s[idx_i] / sum_s;
                }
            } else {
                for idx_i in 0..K {
                    proba[[i, idx_i]] = 1.0 / (K as f64);
                }
            }
        }

        Ok(proba)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_svc_binary() {
        let X = array![[-1.0, -1.0], [-2.0, -1.0], [1.0, 1.0], [2.0, 1.0]];
        let y = array![0.0, 0.0, 1.0, 1.0];

        let mut svc = SVC::new(
            1.0,
            "rbf".to_string(),
            3,
            "scale".to_string(),
            0.0,
            true,
            1e-3,
            1000,
        );

        let res = svc.fit(&X.view(), &y.view());
        assert!(res.is_ok());

        let preds = svc.predict(&X.view()).unwrap();
        assert_eq!(preds, array![0.0, 0.0, 1.0, 1.0]);

        let probas = svc.predict_proba(&X.view()).unwrap();
        assert_eq!(probas.shape(), &[4, 2]);
        assert!(probas[[0, 0]] > 0.5);
        assert!(probas[[1, 0]] > 0.5);
        assert!(probas[[2, 1]] > 0.5);
        assert!(probas[[3, 1]] > 0.5);
    }

    #[test]
    fn test_svc_poly() {
        let X = array![[-1.0, -1.0], [-2.0, -1.0], [1.0, 1.0], [2.0, 1.0]];
        let y = array![0.0, 0.0, 1.0, 1.0];

        let mut svc = SVC::new(
            1.0,
            "poly".to_string(),
            2,
            "scale".to_string(),
            1.0,
            false,
            1e-3,
            1000,
        );

        let res = svc.fit(&X.view(), &y.view());
        assert!(res.is_ok());

        let preds = svc.predict(&X.view()).unwrap();
        assert_eq!(preds, array![0.0, 0.0, 1.0, 1.0]);
    }
}
