#![allow(non_snake_case)]

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

/// Gaussian Naive Bayes
pub struct GaussianNB {
    pub classes_: Option<Vec<f64>>,
    pub class_count_: Option<Array1<f64>>,
    pub class_prior_: Option<Array1<f64>>,
    pub theta_: Option<Array2<f64>>, // means
    pub var_: Option<Array2<f64>>,   // variances
    pub epsilon_: f64,
}

impl GaussianNB {
    pub fn new() -> Self {
        GaussianNB {
            classes_: None,
            class_count_: None,
            class_prior_: None,
            theta_: None,
            var_: None,
            epsilon_: 1e-9, // variance smoothing
        }
    }

    pub fn fit(&mut self, X: &ArrayView2<f64>, y: &ArrayView1<f64>) -> Result<(), String> {
        let n_samples = X.nrows();
        let n_features = X.ncols();
        
        if n_samples != y.len() {
            return Err("X and y sample counts do not match".to_string());
        }

        let mut classes = y.to_vec();
        classes.sort_by(|a, b| a.partial_cmp(b).unwrap());
        classes.dedup();
        
        let n_classes = classes.len();
        let mut class_count = Array1::<f64>::zeros(n_classes);
        let mut theta = Array2::<f64>::zeros((n_classes, n_features));
        let mut var = Array2::<f64>::zeros((n_classes, n_features));

        // Global variance max for epsilon smoothing
        let mut global_var = 0.0;
        for j in 0..n_features {
            let col = X.column(j);
            let mean = col.sum() / (n_samples as f64);
            let mut v = 0.0;
            for &val in col {
                v += (val - mean).powi(2);
            }
            v /= n_samples as f64;
            if v > global_var {
                global_var = v;
            }
        }
        self.epsilon_ = global_var * 1e-9;

        for (c_idx, &cls) in classes.iter().enumerate() {
            let mut count = 0.0;
            for i in 0..n_samples {
                if (y[i] - cls).abs() < f64::EPSILON {
                    count += 1.0;
                    for j in 0..n_features {
                        theta[[c_idx, j]] += X[[i, j]];
                    }
                }
            }
            class_count[c_idx] = count;
            
            for j in 0..n_features {
                theta[[c_idx, j]] /= count;
            }

            for i in 0..n_samples {
                if (y[i] - cls).abs() < f64::EPSILON {
                    for j in 0..n_features {
                        var[[c_idx, j]] += (X[[i, j]] - theta[[c_idx, j]]).powi(2);
                    }
                }
            }
            for j in 0..n_features {
                var[[c_idx, j]] = (var[[c_idx, j]] / count) + self.epsilon_;
            }
        }

        let class_prior = &class_count / (n_samples as f64);

        self.classes_ = Some(classes);
        self.class_count_ = Some(class_count);
        self.class_prior_ = Some(class_prior);
        self.theta_ = Some(theta);
        self.var_ = Some(var);

        Ok(())
    }

    pub fn predict(&self, X: &ArrayView2<f64>) -> Result<Array1<f64>, String> {
        let proba = self.predict_proba(X)?;
        let n_samples = X.nrows();
        let classes = self.classes_.as_ref().unwrap();
        let mut preds = Array1::<f64>::zeros(n_samples);

        for i in 0..n_samples {
            let mut best_idx = 0;
            let mut max_prob = f64::NEG_INFINITY;
            for c in 0..classes.len() {
                if proba[[i, c]] > max_prob {
                    max_prob = proba[[i, c]];
                    best_idx = c;
                }
            }
            preds[i] = classes[best_idx];
        }

        Ok(preds)
    }

    pub fn predict_proba(&self, X: &ArrayView2<f64>) -> Result<Array2<f64>, String> {
        if self.classes_.is_none() {
            return Err("Model is not fitted".to_string());
        }

        let n_samples = X.nrows();
        let classes = self.classes_.as_ref().unwrap();
        let n_classes = classes.len();
        let n_features = X.ncols();
        
        let prior = self.class_prior_.as_ref().unwrap();
        let theta = self.theta_.as_ref().unwrap();
        let var = self.var_.as_ref().unwrap();

        let mut proba = Array2::<f64>::zeros((n_samples, n_classes));

        for i in 0..n_samples {
            let mut log_prob_sum = 0.0;
            let mut max_log_prob = f64::NEG_INFINITY;
            let mut log_probs = vec![0.0; n_classes];

            for c in 0..n_classes {
                let mut log_prob = prior[c].ln();
                for j in 0..n_features {
                    let v = var[[c, j]];
                    let t = theta[[c, j]];
                    let x = X[[i, j]];
                    log_prob -= 0.5 * (2.0 * std::f64::consts::PI * v).ln();
                    log_prob -= 0.5 * (x - t).powi(2) / v;
                }
                log_probs[c] = log_prob;
                if log_prob > max_log_prob {
                    max_log_prob = log_prob;
                }
            }

            // Log-sum-exp trick for numerical stability
            for c in 0..n_classes {
                log_prob_sum += (log_probs[c] - max_log_prob).exp();
            }
            
            for c in 0..n_classes {
                proba[[i, c]] = (log_probs[c] - max_log_prob - log_prob_sum.ln()).exp();
            }
        }

        Ok(proba)
    }
}
