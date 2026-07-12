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

    pub fn partial_fit(&mut self, X: &ArrayView2<f64>, y: &ArrayView1<f64>, classes: Option<Vec<f64>>) -> Result<(), String> {
        let n_samples = X.nrows();
        let n_features = X.ncols();
        
        if n_samples != y.len() {
            return Err("X and y sample counts do not match".to_string());
        }

        // Setup classes if not initialized
        if self.classes_.is_none() {
            if let Some(c) = classes {
                self.classes_ = Some(c);
            } else {
                let mut unique_classes = y.to_vec();
                unique_classes.sort_by(|a, b| a.partial_cmp(b).unwrap());
                unique_classes.dedup();
                self.classes_ = Some(unique_classes);
            }
            let n_classes = self.classes_.as_ref().unwrap().len();
            self.class_count_ = Some(Array1::<f64>::zeros(n_classes));
            self.theta_ = Some(Array2::<f64>::zeros((n_classes, n_features)));
            self.var_ = Some(Array2::<f64>::zeros((n_classes, n_features)));
        }

        let model_classes = self.classes_.as_ref().unwrap().clone();
        let n_classes = model_classes.len();
        
        let mut class_count = self.class_count_.as_ref().unwrap().clone();
        let mut theta = self.theta_.as_ref().unwrap().clone();
        let mut var = self.var_.as_ref().unwrap().clone();

        // Calculate global epsilon
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
        let batch_epsilon = global_var * 1e-9;
        if batch_epsilon > self.epsilon_ {
            self.epsilon_ = batch_epsilon;
        }

        for (c_idx, &cls) in model_classes.iter().enumerate() {
            let mut batch_count = 0.0;
            let mut batch_theta = Array1::<f64>::zeros(n_features);
            
            for i in 0..n_samples {
                if (y[i] - cls).abs() < f64::EPSILON {
                    batch_count += 1.0;
                    for j in 0..n_features {
                        batch_theta[j] += X[[i, j]];
                    }
                }
            }

            if batch_count > 0.0 {
                batch_theta /= batch_count;
                let mut batch_var = Array1::<f64>::zeros(n_features);
                for i in 0..n_samples {
                    if (y[i] - cls).abs() < f64::EPSILON {
                        for j in 0..n_features {
                            batch_var[j] += (X[[i, j]] - batch_theta[j]).powi(2);
                        }
                    }
                }
                batch_var /= batch_count;

                let prev_count = class_count[c_idx];
                let new_count = prev_count + batch_count;

                for j in 0..n_features {
                    let prev_t = theta[[c_idx, j]];
                    let prev_v = var[[c_idx, j]]; // this already has epsilon added, we must remove it for true var
                    
                    // remove epsilon if count > 0
                    let mut raw_prev_v = 0.0;
                    if prev_count > 0.0 {
                        raw_prev_v = prev_v - self.epsilon_;
                    }

                    // Parallel variance update formula
                    let new_t = (prev_count * prev_t + batch_count * batch_theta[j]) / new_count;
                    
                    let s_prev = raw_prev_v * prev_count;
                    let s_batch = batch_var[j] * batch_count;
                    let s_new = s_prev + s_batch + (prev_count * batch_count / new_count) * (prev_t - batch_theta[j]).powi(2);
                    
                    theta[[c_idx, j]] = new_t;
                    var[[c_idx, j]] = (s_new / new_count) + self.epsilon_;
                }
                class_count[c_idx] = new_count;
            } else {
                // Ensure epsilon is updated even if count doesn't change
                if class_count[c_idx] > 0.0 {
                    for j in 0..n_features {
                        var[[c_idx, j]] = (var[[c_idx, j]] - self.epsilon_).max(0.0) + self.epsilon_;
                    }
                }
            }
        }

        let total_count = class_count.sum();
        let class_prior = &class_count / total_count;

        self.class_count_ = Some(class_count);
        self.class_prior_ = Some(class_prior);
        self.theta_ = Some(theta);
        self.var_ = Some(var);

        Ok(())
    }

    pub fn fit(&mut self, X: &ArrayView2<f64>, y: &ArrayView1<f64>) -> Result<(), String> {
        self.classes_ = None;
        self.class_count_ = None;
        self.class_prior_ = None;
        self.theta_ = None;
        self.var_ = None;
        self.epsilon_ = 1e-9;
        self.partial_fit(X, y, None)
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
