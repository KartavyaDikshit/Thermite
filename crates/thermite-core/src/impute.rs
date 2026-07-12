use ndarray::{Array2, ArrayView2};
use crate::linear_model::Ridge;

#[derive(Clone)]
pub struct IterativeImputer {
    pub max_iter: usize,
    pub initial_means: Vec<f64>,
}

impl IterativeImputer {
    pub fn new(max_iter: usize) -> Self {
        Self {
            max_iter,
            initial_means: Vec::new(),
        }
    }

    pub fn fit_transform(&mut self, X: &ArrayView2<f64>) -> Result<Array2<f64>, String> {
        let n = X.nrows();
        let p = X.ncols();
        
        let mut X_imp = Array2::<f64>::zeros((n, p));
        let mut missing_mask = Array2::<bool>::from_elem((n, p), false);
        
        let mut means = vec![0.0; p];
        
        for j in 0..p {
            let mut sum = 0.0;
            let mut count = 0.0;
            for i in 0..n {
                let val = X[[i, j]];
                if val.is_nan() {
                    missing_mask[[i, j]] = true;
                } else {
                    sum += val;
                    count += 1.0;
                }
            }
            means[j] = if count > 0.0 { sum / count } else { 0.0 };
        }
        self.initial_means = means.clone();
        
        for j in 0..p {
            for i in 0..n {
                let val = X[[i, j]];
                X_imp[[i, j]] = if val.is_nan() { means[j] } else { val };
            }
        }
        
        for _iter in 0..self.max_iter {
            for j in 0..p {
                let mut num_missing = 0;
                for i in 0..n {
                    if missing_mask[[i, j]] {
                        num_missing += 1;
                    }
                }
                if num_missing == 0 {
                    continue;
                }
                
                let n_train = n - num_missing;
                if n_train == 0 {
                    continue;
                }
                
                let mut X_train = Array2::<f64>::zeros((n_train, p - 1));
                let mut y_train = ndarray::Array1::<f64>::zeros(n_train);
                
                let mut X_test = Array2::<f64>::zeros((num_missing, p - 1));
                
                let mut train_idx = 0;
                let mut test_idx = 0;
                
                for i in 0..n {
                    if missing_mask[[i, j]] {
                        let mut col_idx = 0;
                        for c in 0..p {
                            if c != j {
                                X_test[[test_idx, col_idx]] = X_imp[[i, c]];
                                col_idx += 1;
                            }
                        }
                        test_idx += 1;
                    } else {
                        y_train[train_idx] = X_imp[[i, j]];
                        let mut col_idx = 0;
                        for c in 0..p {
                            if c != j {
                                X_train[[train_idx, col_idx]] = X_imp[[i, c]];
                                col_idx += 1;
                            }
                        }
                        train_idx += 1;
                    }
                }
                
                let mut model = Ridge::new(1.0, true);
                model.fit(&X_train.view(), &y_train.view())?;
                
                let preds = model.predict(&X_test.view())?;
                
                let mut p_idx = 0;
                for i in 0..n {
                    if missing_mask[[i, j]] {
                        X_imp[[i, j]] = preds[p_idx];
                        p_idx += 1;
                    }
                }
            }
        }
        
        Ok(X_imp)
    }
}
