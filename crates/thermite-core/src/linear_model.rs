#![allow(non_snake_case)]

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rayon::prelude::*;
use sprs::CsMat;

use crate::metrics;

// ==========================================
// Helper: check for non-finite values
// ==========================================
fn check_finite_2d(X: &ArrayView2<f64>) -> Result<(), String> {
    if X.iter().any(|&val| !val.is_finite()) {
        return Err("Input contains NaN or infinity values".to_string());
    }
    Ok(())
}

fn check_finite_1d(y: &ArrayView1<f64>) -> Result<(), String> {
    if y.iter().any(|&val| !val.is_finite()) {
        return Err("Input contains NaN or infinity values".to_string());
    }
    Ok(())
}

// ==========================================
// Helper: Gaussian elimination to solve A * x = b
// where A is (n x n) and b is (n x m).
// Returns x as (n x m).
// ==========================================
fn solve_linear_system(A: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>, String> {
    let n = A.nrows();
    if n != A.ncols() {
        return Err("Matrix A must be square".to_string());
    }
    if n != b.nrows() {
        return Err("Dimension mismatch between A and b".to_string());
    }
    let m = b.ncols();

    // Build augmented matrix [A | b] of shape (n, n+m)
    let mut aug = Array2::<f64>::zeros((n, n + m));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = A[[i, j]];
        }
        for j in 0..m {
            aug[[i, n + j]] = b[[i, j]];
        }
    }

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        let mut max_val = aug[[col, col]].abs();
        for row in (col + 1)..n {
            let val = aug[[row, col]].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }
        if max_val < 1e-14 {
            return Err("Matrix is singular or near-singular".to_string());
        }
        // Swap rows
        if max_row != col {
            for j in 0..(n + m) {
                let tmp = aug[[col, j]];
                aug[[col, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = tmp;
            }
        }
        // Eliminate below
        let pivot = aug[[col, col]];
        for row in (col + 1)..n {
            let factor = aug[[row, col]] / pivot;
            aug[[row, col]] = 0.0;
            for j in (col + 1)..(n + m) {
                aug[[row, j]] -= factor * aug[[col, j]];
            }
        }
    }

    // Back substitution
    let mut x = Array2::<f64>::zeros((n, m));
    for col in (0..n).rev() {
        let pivot = aug[[col, col]];
        for j in 0..m {
            let mut val = aug[[col, n + j]];
            for k in (col + 1)..n {
                val -= aug[[col, k]] * x[[k, j]];
            }
            x[[col, j]] = val / pivot;
        }
    }

    Ok(x)
}

// ==========================================
// Helper: prepend a column of ones to X (for intercept)
// ==========================================
fn add_intercept_column(X: &ArrayView2<f64>) -> Array2<f64> {
    let n = X.nrows();
    let p = X.ncols();
    let mut X_aug = Array2::<f64>::ones((n, p + 1));
    for i in 0..n {
        for j in 0..p {
            X_aug[[i, j + 1]] = X[[i, j]];
        }
    }
    X_aug
}

// ==========================================
// Helper: mat-mul  (A^T * B)
// ==========================================
fn at_b(A: &Array2<f64>, B: &Array2<f64>) -> Array2<f64> {
    A.t().dot(B)
}

// ==========================================
// LinearRegression  OLS via normal equation
// ==========================================
pub struct LinearRegression {
    pub coef_: Option<Array1<f64>>,
    pub intercept_: f64,
    pub fit_intercept: bool,
}

impl LinearRegression {
    pub fn new(fit_intercept: bool) -> Self {
        LinearRegression {
            coef_: None,
            intercept_: 0.0,
            fit_intercept,
        }
    }

    /// Fit using the normal equation: (X^T X)^{-1} X^T y
    pub fn fit(&mut self, X: &ArrayView2<f64>, y: &ArrayView1<f64>) -> Result<(), String> {
        if X.nrows() == 0 || X.ncols() == 0 {
            return Err("Input array is empty".to_string());
        }
        if X.nrows() != y.len() {
            return Err(format!(
                "Sample count mismatch: X has {} rows, y has {} elements",
                X.nrows(),
                y.len()
            ));
        }
        check_finite_2d(X)?;
        check_finite_1d(y)?;

        let X_work = if self.fit_intercept {
            add_intercept_column(X)
        } else {
            X.to_owned()
        };

        // y as column vector (n, 1)
        let y_col = y.to_owned().into_shape((y.len(), 1))
            .map_err(|e| format!("Failed to reshape y: {}", e))?;

        // X^T X
        let XtX = at_b(&X_work, &X_work);
        // X^T y
        let Xty = at_b(&X_work, &y_col);

        // Solve (X^T X) w = X^T y
        let w = solve_linear_system(&XtX, &Xty)?;

        if self.fit_intercept {
            self.intercept_ = w[[0, 0]];
            let p = X.ncols();
            let mut coef = Array1::<f64>::zeros(p);
            for j in 0..p {
                coef[j] = w[[j + 1, 0]];
            }
            self.coef_ = Some(coef);
        } else {
            self.intercept_ = 0.0;
            let p = X.ncols();
            let mut coef = Array1::<f64>::zeros(p);
            for j in 0..p {
                coef[j] = w[[j, 0]];
            }
            self.coef_ = Some(coef);
        }

        Ok(())
    }

    pub fn predict(&self, X: &ArrayView2<f64>) -> Result<Array1<f64>, String> {
        let coef = self.coef_.as_ref().ok_or("Model is not fitted yet")?;
        if X.ncols() != coef.len() {
            return Err(format!(
                "Feature mismatch: expected {}, got {}",
                coef.len(),
                X.ncols()
            ));
        }
        check_finite_2d(X)?;

        let intercept = self.intercept_;
        let preds = X.dot(coef) + intercept;

        Ok(preds)
    }

    /// R² score
    pub fn score(&self, X: &ArrayView2<f64>, y: &ArrayView1<f64>) -> Result<f64, String> {
        let preds = self.predict(X)?;
        metrics::r2_score(y.as_slice().unwrap(), preds.as_slice().unwrap())
    }
}

// ==========================================
// Ridge  OLS + L2 penalty
// ==========================================
pub struct Ridge {
    pub coef_: Option<Array1<f64>>,
    pub intercept_: f64,
    pub fit_intercept: bool,
    pub alpha: f64,
}

impl Ridge {
    pub fn new(alpha: f64, fit_intercept: bool) -> Self {
        Ridge {
            coef_: None,
            intercept_: 0.0,
            fit_intercept,
            alpha,
        }
    }

    /// Fit using: (X^T X + alpha * I)^{-1} X^T y
    pub fn fit(&mut self, X: &ArrayView2<f64>, y: &ArrayView1<f64>) -> Result<(), String> {
        if X.nrows() == 0 || X.ncols() == 0 {
            return Err("Input array is empty".to_string());
        }
        if X.nrows() != y.len() {
            return Err(format!(
                "Sample count mismatch: X has {} rows, y has {} elements",
                X.nrows(),
                y.len()
            ));
        }
        check_finite_2d(X)?;
        check_finite_1d(y)?;

        let X_work = if self.fit_intercept {
            add_intercept_column(X)
        } else {
            X.to_owned()
        };

        let y_col = y.to_owned().into_shape((y.len(), 1))
            .map_err(|e| format!("Failed to reshape y: {}", e))?;

        let mut XtX = at_b(&X_work, &X_work);
        let p = XtX.nrows();

        // Add alpha * I (skip intercept column if fit_intercept)
        let start = if self.fit_intercept { 1 } else { 0 };
        for i in start..p {
            XtX[[i, i]] += self.alpha;
        }

        let Xty = at_b(&X_work, &y_col);
        let w = solve_linear_system(&XtX, &Xty)?;

        if self.fit_intercept {
            self.intercept_ = w[[0, 0]];
            let n_features = X.ncols();
            let mut coef = Array1::<f64>::zeros(n_features);
            for j in 0..n_features {
                coef[j] = w[[j + 1, 0]];
            }
            self.coef_ = Some(coef);
        } else {
            self.intercept_ = 0.0;
            let n_features = X.ncols();
            let mut coef = Array1::<f64>::zeros(n_features);
            for j in 0..n_features {
                coef[j] = w[[j, 0]];
            }
            self.coef_ = Some(coef);
        }

        Ok(())
    }

    pub fn predict(&self, X: &ArrayView2<f64>) -> Result<Array1<f64>, String> {
        let coef = self.coef_.as_ref().ok_or("Model is not fitted yet")?;
        if X.ncols() != coef.len() {
            return Err(format!(
                "Feature mismatch: expected {}, got {}",
                coef.len(),
                X.ncols()
            ));
        }
        check_finite_2d(X)?;

        let intercept = self.intercept_;
        let preds: Vec<f64> = X
            .axis_iter(Axis(0))
            .into_par_iter()
            .map(|row| {
                let mut val = intercept;
                for j in 0..row.len() {
                    val += row[j] * coef[j];
                }
                val
            })
            .collect();

        Ok(Array1::from(preds))
    }

    pub fn score(&self, X: &ArrayView2<f64>, y: &ArrayView1<f64>) -> Result<f64, String> {
        let preds = self.predict(X)?;
        metrics::r2_score(y.as_slice().unwrap(), preds.as_slice().unwrap())
    }
}

// ==========================================
// Lasso  Coordinate Descent with L1 penalty
// ==========================================
pub struct Lasso {
    pub coef_: Option<Array1<f64>>,
    pub intercept_: f64,
    pub fit_intercept: bool,
    pub alpha: f64,
    pub max_iter: usize,
    pub tol: f64,
}

impl Lasso {
    pub fn new(alpha: f64, fit_intercept: bool, max_iter: usize, tol: f64) -> Self {
        Lasso {
            coef_: None,
            intercept_: 0.0,
            fit_intercept,
            alpha,
            max_iter,
            tol,
        }
    }

    /// Soft-thresholding operator
    #[inline]
    fn soft_threshold(val: f64, lambda: f64) -> f64 {
        if val > lambda {
            val - lambda
        } else if val < -lambda {
            val + lambda
        } else {
            0.0
        }
    }

    /// Fit using coordinate descent
    pub fn fit(&mut self, X: &ArrayView2<f64>, y: &ArrayView1<f64>) -> Result<(), String> {
        if X.nrows() == 0 || X.ncols() == 0 {
            return Err("Input array is empty".to_string());
        }
        if X.nrows() != y.len() {
            return Err(format!(
                "Sample count mismatch: X has {} rows, y has {} elements",
                X.nrows(),
                y.len()
            ));
        }
        check_finite_2d(X)?;
        check_finite_1d(y)?;

        let n = X.nrows();
        let p = X.ncols();
        let n_f64 = n as f64;

        // Center the data if fit_intercept
        let (X_work, y_work, x_mean, y_mean) = if self.fit_intercept {
            let x_mean: Array1<f64> = X.mean_axis(Axis(0)).unwrap();
            let y_mean = y.mean().unwrap();

            let mut X_centered = X.to_owned();
            X_centered -= &x_mean;
            
            let y_centered: Array1<f64> = y.mapv(|v| v - y_mean);
            (X_centered, y_centered, x_mean, y_mean)
        } else {
            let x_mean = Array1::<f64>::zeros(p);
            (X.to_owned(), y.to_owned(), x_mean, 0.0)
        };

        // Precompute column norms squared
        let col_norms_sq: Vec<f64> = (0..p)
            .into_par_iter()
            .map(|j| {
                X_work.column(j).dot(&X_work.column(j))
            })
            .collect();

        let mut coef = Array1::<f64>::zeros(p);
        let mut residual = y_work.clone();

        let lambda = self.alpha * n_f64;

        for _iter in 0..self.max_iter {
            let mut max_change = 0.0_f64;

            for j in 0..p {
                let old_coef = coef[j];

                // Compute partial residual correlation
                let rho = X_work.column(j).dot(&residual) + old_coef * col_norms_sq[j];

                let new_coef = if col_norms_sq[j] == 0.0 {
                    0.0
                } else {
                    Self::soft_threshold(rho, lambda) / col_norms_sq[j]
                };

                let delta = new_coef - old_coef;
                if delta.abs() > 0.0 {
                    // Update residual
                    ndarray::Zip::from(&mut residual).and(&X_work.column(j)).for_each(|r, &x| *r -= delta * x);
                    coef[j] = new_coef;
                    max_change = max_change.max(delta.abs());
                }
            }

            if max_change < self.tol {
                break;
            }
        }

        if self.fit_intercept {
            let mut intercept = y_mean;
            for j in 0..p {
                intercept -= coef[j] * x_mean[j];
            }
            self.intercept_ = intercept;
        } else {
            self.intercept_ = 0.0;
        }

        self.coef_ = Some(coef);
        Ok(())
    }

    pub fn predict(&self, X: &ArrayView2<f64>) -> Result<Array1<f64>, String> {
        let coef = self.coef_.as_ref().ok_or("Model is not fitted yet")?;
        if X.ncols() != coef.len() {
            return Err(format!(
                "Feature mismatch: expected {}, got {}",
                coef.len(),
                X.ncols()
            ));
        }
        check_finite_2d(X)?;

        let intercept = self.intercept_;
        let preds: Vec<f64> = X
            .axis_iter(Axis(0))
            .into_par_iter()
            .map(|row| {
                let mut val = intercept;
                for j in 0..row.len() {
                    val += row[j] * coef[j];
                }
                val
            })
            .collect();

        Ok(Array1::from(preds))
    }

    pub fn score(&self, X: &ArrayView2<f64>, y: &ArrayView1<f64>) -> Result<f64, String> {
        let preds = self.predict(X)?;
        metrics::r2_score(y.as_slice().unwrap(), preds.as_slice().unwrap())
    }
}

// ==========================================
// LogisticRegression  gradient descent
// Binary / multiclass (one-vs-rest)
// ==========================================
pub struct LogisticRegression {
    /// Inverse of regularization strength (higher = weaker regularization)
    pub C: f64,
    pub max_iter: usize,
    pub tol: f64,
    pub penalty: String,
    /// For binary: shape (n_features,). For multiclass: Vec of per-class weight vectors.
    pub coef_: Option<Array2<f64>>,
    /// Per-class intercept
    pub intercept_: Option<Array1<f64>>,
    /// Ordered unique classes
    pub classes_: Option<Vec<f64>>,
    pub learning_rate: f64,
}

impl LogisticRegression {
    pub fn new(C: f64, max_iter: usize, tol: f64, penalty: &str) -> Self {
        LogisticRegression {
            C,
            max_iter,
            tol,
            penalty: penalty.to_string(),
            coef_: None,
            intercept_: None,
            classes_: None,
            learning_rate: 0.1,
        }
    }

    #[inline]
    fn sigmoid(z: f64) -> f64 {
        1.0 / (1.0 + (-z).exp())
    }

    /// Fit binary logistic regression using L-BFGS.
    /// Uses limited-memory BFGS (m=10) for fast convergence with only gradient evaluations.
    /// Returns (weights, bias).
    fn fit_binary(
        &self,
        X: &Array2<f64>,
        y_binary: &Array1<f64>, // 0/1 labels
        n_features: usize,
    ) -> Result<(Array1<f64>, f64), String> {
        let n = X.nrows();
        let n_f64 = n as f64;
        let lambda = 1.0 / self.C;

        let p = n_features + 1;
        let mut w = Array1::<f64>::zeros(p);

        // L-BFGS storage (limited memory, m = 10)
        let m: usize = 10;
        let mut s_hist: Vec<Array1<f64>> = Vec::with_capacity(m);
        let mut y_hist: Vec<Array1<f64>> = Vec::with_capacity(m);
        let mut rho_hist: Vec<f64> = Vec::with_capacity(m);

        // Compute initial gradient
        let compute_grad = |w: &Array1<f64>| -> Array1<f64> {
            let bias = w[0];
            let w_feat = w.slice(ndarray::s![1..]);
            let z = X.dot(&w_feat) + bias;
            let h = z.mapv(Self::sigmoid);
            let diff = &h - y_binary;
            
            let mut grad = Array1::<f64>::zeros(p);
            grad[0] = diff.sum() / n_f64;
            
            let mut grad_feat = X.t().dot(&diff);
            grad_feat /= n_f64;
            
            for j in 0..n_features {
                grad_feat[j] += lambda * w_feat[j];
            }
            
            grad.slice_mut(ndarray::s![1..]).assign(&grad_feat);
            grad
        };

        let compute_loss = |w: &Array1<f64>| -> f64 {
            let bias = w[0];
            let w_feat = w.slice(ndarray::s![1..]);
            let z = X.dot(&w_feat) + bias;
            let mut loss = 0.0;
            for i in 0..n {
                let zi = z[i];
                if zi >= 0.0 {
                    loss += (1.0 + (-zi).exp()).ln() - y_binary[i] * zi;
                } else {
                    loss += zi.exp().ln_1p() - y_binary[i] * zi;
                }
            }
            loss /= n_f64;
            for j in 0..n_features {
                loss += 0.5 * lambda * w_feat[j] * w_feat[j];
            }
            loss
        };

        let mut grad = compute_grad(&w);
        let mut prev_grad = grad.clone();

        for _iter in 0..self.max_iter {
            // L-BFGS two-loop recursion to compute search direction
            let mut q = grad.clone();
            let hist_len = s_hist.len();
            let mut alpha_vec = vec![0.0; hist_len];

            // First loop (backward)
            for i in (0..hist_len).rev() {
                alpha_vec[i] = rho_hist[i] * s_hist[i].dot(&q);
                q = q - alpha_vec[i] * &y_hist[i];
            }

            // Initial Hessian approximation: H0 = gamma * I
            let gamma = if hist_len > 0 {
                let last = hist_len - 1;
                s_hist[last].dot(&y_hist[last]) / y_hist[last].dot(&y_hist[last])
            } else {
                1.0
            };
            let mut r = q * gamma;

            // Second loop (forward)
            for i in 0..hist_len {
                let beta = rho_hist[i] * y_hist[i].dot(&r);
                r = r + (alpha_vec[i] - beta) * &s_hist[i];
            }

            // Search direction (negative because we minimize)
            let direction = -&r;

            // Backtracking line search (Armijo condition)
            let grad_dot_dir = grad.dot(&direction);
            if grad_dot_dir >= 0.0 {
                // Reset if not a descent direction
                s_hist.clear();
                y_hist.clear();
                rho_hist.clear();
                let lr = self.learning_rate;
                w = &w - &(&grad * lr);
                grad = compute_grad(&w);
                prev_grad = grad.clone();
                continue;
            }

            let mut step = 1.0;
            let c1 = 1e-4;
            let current_loss = compute_loss(&w);
            for _ in 0..20 {
                let w_new = &w + &(&direction * step);
                let new_loss = compute_loss(&w_new);
                if new_loss <= current_loss + c1 * step * grad_dot_dir {
                    break;
                }
                step *= 0.5;
            }

            let s = &direction * step;
            let w_new = &w + &s;
            let new_grad = compute_grad(&w_new);
            let y_diff = &new_grad - &grad;
            let sy = s.dot(&y_diff);

            if sy > 1e-10 {
                if s_hist.len() >= m {
                    s_hist.remove(0);
                    y_hist.remove(0);
                    rho_hist.remove(0);
                }
                s_hist.push(s.clone());
                y_hist.push(y_diff);
                rho_hist.push(1.0 / sy);
            }

            // Check convergence
            let max_change = s.iter().map(|v| v.abs()).fold(0.0_f64, |a, b| a.max(b));
            w = w_new;
            prev_grad = grad.clone();
            grad = new_grad;

            if max_change < self.tol {
                break;
            }
        }

        let bias = w[0];
        let weights = w.slice(ndarray::s![1..]).to_owned();

        Ok((weights, bias))
    }

    /// Fit binary logistic regression using L-BFGS on sparse matrix.
    fn fit_binary_sparse(
        &self,
        X: &CsMat<f64>,
        y_binary: &Array1<f64>, // 0/1 labels
        n_features: usize,
    ) -> Result<(Array1<f64>, f64), String> {
        let n = X.rows();
        let n_f64 = n as f64;
        let lambda = 1.0 / self.C;

        let p = n_features + 1;
        let mut w = Array1::<f64>::zeros(p);

        // L-BFGS storage (limited memory, m = 10)
        let m: usize = 10;
        let mut s_hist: Vec<Array1<f64>> = Vec::with_capacity(m);
        let mut y_hist: Vec<Array1<f64>> = Vec::with_capacity(m);
        let mut rho_hist: Vec<f64> = Vec::with_capacity(m);

        // Compute initial gradient
        let compute_grad = |w: &Array1<f64>| -> Array1<f64> {
            let mut grad = Array1::<f64>::zeros(p);
            
            for (i, vec) in X.outer_iterator().enumerate() {
                let mut z = w[0];
                for (col_idx, &val) in vec.iter() {
                    z += val * w[col_idx + 1];
                }
                
                let h = Self::sigmoid(z);
                let diff = h - y_binary[i];
                
                grad[0] += diff;
                for (col_idx, &val) in vec.iter() {
                    grad[col_idx + 1] += diff * val;
                }
            }
            
            grad /= n_f64;
            // L2 regularization on weights only (skip bias at index 0)
            for j in 1..p {
                grad[j] += lambda * w[j];
            }
            grad
        };

        let compute_loss = |w: &Array1<f64>| -> f64 {
            let mut loss = 0.0;
            for (i, vec) in X.outer_iterator().enumerate() {
                let mut zi = w[0];
                for (col_idx, &val) in vec.iter() {
                    zi += val * w[col_idx + 1];
                }
                // Numerically stable log-loss
                if zi >= 0.0 {
                    loss += (1.0 + (-zi).exp()).ln() - y_binary[i] * zi;
                } else {
                    loss += zi.exp().ln_1p() - y_binary[i] * zi;
                }
            }
            loss /= n_f64;
            // L2 regularization
            for j in 1..p {
                loss += 0.5 * lambda * w[j] * w[j];
            }
            loss
        };

        let mut grad = compute_grad(&w);

        for _iter in 0..self.max_iter {
            let mut q = grad.clone();
            let hist_len = s_hist.len();
            let mut alpha_vec = vec![0.0; hist_len];

            for i in (0..hist_len).rev() {
                alpha_vec[i] = rho_hist[i] * s_hist[i].dot(&q);
                q = q - alpha_vec[i] * &y_hist[i];
            }

            let gamma = if hist_len > 0 {
                let last = hist_len - 1;
                s_hist[last].dot(&y_hist[last]) / y_hist[last].dot(&y_hist[last])
            } else {
                1.0
            };
            let mut r = q * gamma;

            for i in 0..hist_len {
                let beta = rho_hist[i] * y_hist[i].dot(&r);
                r = r + (alpha_vec[i] - beta) * &s_hist[i];
            }

            let direction = -&r;
            let grad_dot_dir = grad.dot(&direction);
            
            if grad_dot_dir >= 0.0 {
                s_hist.clear();
                y_hist.clear();
                rho_hist.clear();
                let lr = self.learning_rate;
                w = &w - &(&grad * lr);
                grad = compute_grad(&w);
                continue;
            }

            let mut step = 1.0;
            let c1 = 1e-4;
            let current_loss = compute_loss(&w);
            for _ in 0..20 {
                let w_new = &w + &(&direction * step);
                let new_loss = compute_loss(&w_new);
                if new_loss <= current_loss + c1 * step * grad_dot_dir {
                    break;
                }
                step *= 0.5;
            }

            let s = &direction * step;
            let w_new = &w + &s;
            let new_grad = compute_grad(&w_new);
            let y_diff = &new_grad - &grad;
            let sy = s.dot(&y_diff);

            if sy > 1e-10 {
                if s_hist.len() >= m {
                    s_hist.remove(0);
                    y_hist.remove(0);
                    rho_hist.remove(0);
                }
                s_hist.push(s.clone());
                y_hist.push(y_diff);
                rho_hist.push(1.0 / sy);
            }

            let max_change = s.iter().map(|v| v.abs()).fold(0.0_f64, |a, b| a.max(b));
            w = w_new;
            grad = new_grad;

            if max_change < self.tol {
                break;
            }
        }

        let bias = w[0];
        let weights = w.slice(ndarray::s![1..]).to_owned();

        Ok((weights, bias))
    }

    /// Fit the model on sparse matrix
    pub fn fit_sparse(&mut self, X: &CsMat<f64>, y: &ArrayView1<f64>) -> Result<(), String> {
        if X.rows() == 0 || X.cols() == 0 {
            return Err("Input array is empty".to_string());
        }
        if X.rows() != y.len() {
            return Err(format!(
                "Sample count mismatch: X has {} rows, y has {} elements",
                X.rows(),
                y.len()
            ));
        }
        check_finite_1d(y)?;

        if self.penalty != "l2" {
            return Err(format!("Unsupported penalty '{}'.", self.penalty));
        }

        let mut classes: Vec<f64> = Vec::new();
        for &v in y.iter() {
            if !classes.iter().any(|&c| (c - v).abs() < f64::EPSILON) {
                classes.push(v);
            }
        }
        classes.sort_unstable_by(|a, b| a.total_cmp(b));

        let n_features = X.cols();

        if classes.len() == 2 {
            let pos = classes[1];
            let y_binary = Array1::from(
                y.iter()
                    .map(|&v| if (v - pos).abs() < f64::EPSILON { 1.0 } else { 0.0 })
                    .collect::<Vec<f64>>(),
            );
            let (w, b) = self.fit_binary_sparse(X, &y_binary, n_features)?;
            let mut coef = Array2::<f64>::zeros((1, n_features));
            for j in 0..n_features {
                coef[[0, j]] = w[j];
            }
            self.coef_ = Some(coef);
            self.intercept_ = Some(Array1::from(vec![b]));
        } else {
            let n_classes = classes.len();
            let mut coef = Array2::<f64>::zeros((n_classes, n_features));
            let mut intercept = Array1::<f64>::zeros(n_classes);

            for (c_idx, &cls) in classes.iter().enumerate() {
                let y_binary = Array1::from(
                    y.iter()
                        .map(|&v| if (v - cls).abs() < f64::EPSILON { 1.0 } else { 0.0 })
                        .collect::<Vec<f64>>(),
                );
                let (w, b) = self.fit_binary_sparse(X, &y_binary, n_features)?;
                for j in 0..n_features {
                    coef[[c_idx, j]] = w[j];
                }
                intercept[c_idx] = b;
            }
            self.coef_ = Some(coef);
            self.intercept_ = Some(intercept);
        }

        self.classes_ = Some(classes);
        Ok(())
    }

    /// Fit the model. Supports binary and multiclass (one-vs-rest).
    pub fn fit(&mut self, X: &ArrayView2<f64>, y: &ArrayView1<f64>) -> Result<(), String> {
        if X.nrows() == 0 || X.ncols() == 0 {
            return Err("Input array is empty".to_string());
        }
        if X.nrows() != y.len() {
            return Err(format!(
                "Sample count mismatch: X has {} rows, y has {} elements",
                X.nrows(),
                y.len()
            ));
        }
        check_finite_2d(X)?;
        check_finite_1d(y)?;

        if self.penalty != "l2" {
            return Err(format!(
                "Unsupported penalty '{}'. Only 'l2' is supported.",
                self.penalty
            ));
        }

        // Discover classes
        let mut classes: Vec<f64> = Vec::new();
        for &v in y.iter() {
            if !classes.iter().any(|&c| (c - v).abs() < f64::EPSILON) {
                classes.push(v);
            }
        }
        classes.sort_unstable_by(|a, b| a.total_cmp(b));

        let n_features = X.ncols();
        let X_owned = X.to_owned();

        if classes.len() == 2 {
            // Binary: map to 0/1 where positive class is classes[1]
            let pos = classes[1];
            let y_binary = Array1::from(
                y.iter()
                    .map(|&v| if (v - pos).abs() < f64::EPSILON { 1.0 } else { 0.0 })
                    .collect::<Vec<f64>>(),
            );
            let (w, b) = self.fit_binary(&X_owned, &y_binary, n_features)?;
            let mut coef = Array2::<f64>::zeros((1, n_features));
            for j in 0..n_features {
                coef[[0, j]] = w[j];
            }
            self.coef_ = Some(coef);
            self.intercept_ = Some(Array1::from(vec![b]));
        } else {
            // Multiclass: one-vs-rest
            let n_classes = classes.len();
            let mut coef = Array2::<f64>::zeros((n_classes, n_features));
            let mut intercept = Array1::<f64>::zeros(n_classes);

            for (c_idx, &cls) in classes.iter().enumerate() {
                let y_binary = Array1::from(
                    y.iter()
                        .map(|&v| if (v - cls).abs() < f64::EPSILON { 1.0 } else { 0.0 })
                        .collect::<Vec<f64>>(),
                );
                let (w, b) = self.fit_binary(&X_owned, &y_binary, n_features)?;
                for j in 0..n_features {
                    coef[[c_idx, j]] = w[j];
                }
                intercept[c_idx] = b;
            }
            self.coef_ = Some(coef);
            self.intercept_ = Some(intercept);
        }

        self.classes_ = Some(classes);
        Ok(())
    }

    pub fn partial_fit(&mut self, X: &ArrayView2<f64>, y: &ArrayView1<f64>, classes_opt: Option<Vec<f64>>) -> Result<(), String> {
        if X.nrows() == 0 || X.ncols() == 0 {
            return Err("Input array is empty".to_string());
        }
        if X.nrows() != y.len() {
            return Err("Sample count mismatch".to_string());
        }
        check_finite_2d(X)?;
        check_finite_1d(y)?;

        let n_features = X.ncols();
        let lambda = 1.0 / self.C;
        let lr = self.learning_rate;
        let n_f64 = X.nrows() as f64;

        if self.classes_.is_none() {
            if let Some(c) = classes_opt {
                self.classes_ = Some(c);
            } else {
                let mut c: Vec<f64> = y.iter().cloned().collect();
                c.sort_unstable_by(|a, b| a.total_cmp(b));
                c.dedup();
                self.classes_ = Some(c);
            }
            let classes = self.classes_.as_ref().unwrap();
            let n_classes = classes.len();
            if n_classes == 2 {
                self.coef_ = Some(Array2::<f64>::zeros((1, n_features)));
                self.intercept_ = Some(Array1::<f64>::zeros(1));
            } else {
                self.coef_ = Some(Array2::<f64>::zeros((n_classes, n_features)));
                self.intercept_ = Some(Array1::<f64>::zeros(n_classes));
            }
        }

        let classes = self.classes_.as_ref().unwrap();
        let mut coef = self.coef_.as_ref().unwrap().clone();
        let mut intercept = self.intercept_.as_ref().unwrap().clone();

        if classes.len() == 2 {
            let pos = classes[1];
            let y_binary = Array1::from(
                y.iter()
                    .map(|&v| if (v - pos).abs() < f64::EPSILON { 1.0 } else { 0.0 })
                    .collect::<Vec<f64>>(),
            );
            
            let mut w_feat = coef.row(0).to_owned();
            let mut bias = intercept[0];

            let z = X.dot(&w_feat) + bias;
            let h = z.mapv(Self::sigmoid);
            let diff = &h - &y_binary;

            bias -= lr * (diff.sum() / n_f64);
            
            let mut grad_feat = X.t().dot(&diff);
            grad_feat /= n_f64;
            
            for j in 0..n_features {
                grad_feat[j] += lambda * w_feat[j];
                w_feat[j] -= lr * grad_feat[j];
            }
            
            coef.row_mut(0).assign(&w_feat);
            intercept[0] = bias;

        } else {
            let n_classes = classes.len();
            for c_idx in 0..n_classes {
                let cls = classes[c_idx];
                let y_binary = Array1::from(
                    y.iter()
                        .map(|&v| if (v - cls).abs() < f64::EPSILON { 1.0 } else { 0.0 })
                        .collect::<Vec<f64>>(),
                );
                
                let mut w_feat = coef.row(c_idx).to_owned();
                let mut bias = intercept[c_idx];

                let z = X.dot(&w_feat) + bias;
                let h = z.mapv(Self::sigmoid);
                let diff = &h - &y_binary;

                bias -= lr * (diff.sum() / n_f64);
                
                let mut grad_feat = X.t().dot(&diff);
                grad_feat /= n_f64;
                
                for j in 0..n_features {
                    grad_feat[j] += lambda * w_feat[j];
                    w_feat[j] -= lr * grad_feat[j];
                }
                
                coef.row_mut(c_idx).assign(&w_feat);
                intercept[c_idx] = bias;
            }
        }

        self.coef_ = Some(coef);
        self.intercept_ = Some(intercept);

        Ok(())
    }

    /// Predict class labels.
    pub fn predict(&self, X: &ArrayView2<f64>) -> Result<Array1<f64>, String> {
        let coef = self.coef_.as_ref().ok_or("Model is not fitted yet")?;
        let intercept = self.intercept_.as_ref().unwrap();
        let classes = self.classes_.as_ref().unwrap();
        check_finite_2d(X)?;

        let n_features = coef.ncols();
        if X.ncols() != n_features {
            return Err(format!(
                "Feature mismatch: expected {}, got {}",
                n_features,
                X.ncols()
            ));
        }

        let n = X.nrows();

        if classes.len() == 2 {
            // Binary
            let preds: Vec<f64> = X
                .axis_iter(Axis(0))
                .into_par_iter()
                .map(|row| {
                    let mut z = intercept[0];
                    for j in 0..n_features {
                        z += row[j] * coef[[0, j]];
                    }
                    if Self::sigmoid(z) >= 0.5 { classes[1] } else { classes[0] }
                })
                .collect();
            Ok(Array1::from(preds))
        } else {
            // Multiclass: argmax of OVR scores
            let n_classes = classes.len();
            let preds: Vec<f64> = X
                .axis_iter(Axis(0))
                .into_par_iter()
                .map(|row| {
                    let mut best_score = f64::NEG_INFINITY;
                    let mut best_class = classes[0];
                    for c in 0..n_classes {
                        let mut z = intercept[c];
                        for j in 0..n_features {
                            z += row[j] * coef[[c, j]];
                        }
                        if z > best_score {
                            best_score = z;
                            best_class = classes[c];
                        }
                    }
                    best_class
                })
                .collect();
            Ok(Array1::from(preds))
        }
    }

    /// Predict probabilities. Returns (n_samples, n_classes) array.
    pub fn predict_proba(&self, X: &ArrayView2<f64>) -> Result<Array2<f64>, String> {
        let coef = self.coef_.as_ref().ok_or("Model is not fitted yet")?;
        let intercept = self.intercept_.as_ref().unwrap();
        let classes = self.classes_.as_ref().unwrap();
        check_finite_2d(X)?;

        let n_features = coef.ncols();
        if X.ncols() != n_features {
            return Err(format!(
                "Feature mismatch: expected {}, got {}",
                n_features,
                X.ncols()
            ));
        }

        let n = X.nrows();
        let n_classes = classes.len();

        if n_classes == 2 {
            let mut proba = Array2::<f64>::zeros((n, 2));
            for i in 0..n {
                let mut z = intercept[0];
                for j in 0..n_features {
                    z += X[[i, j]] * coef[[0, j]];
                }
                let p1 = Self::sigmoid(z);
                proba[[i, 0]] = 1.0 - p1;
                proba[[i, 1]] = p1;
            }
            Ok(proba)
        } else {
            // OVR probabilities, then normalize per row
            let mut proba = Array2::<f64>::zeros((n, n_classes));
            for i in 0..n {
                let mut row_sum = 0.0;
                for c in 0..n_classes {
                    let mut z = intercept[c];
                    for j in 0..n_features {
                        z += X[[i, j]] * coef[[c, j]];
                    }
                    let p = Self::sigmoid(z);
                    proba[[i, c]] = p;
                    row_sum += p;
                }
                // Normalize
                if row_sum > 0.0 {
                    for c in 0..n_classes {
                        proba[[i, c]] /= row_sum;
                    }
                }
            }
            Ok(proba)
        }
    }

    /// Predict class labels for sparse matrix.
    pub fn predict_sparse(&self, X: &CsMat<f64>) -> Result<Array1<f64>, String> {
        let coef = self.coef_.as_ref().ok_or("Model is not fitted yet")?;
        let intercept = self.intercept_.as_ref().unwrap();
        let classes = self.classes_.as_ref().unwrap();

        let n_features = coef.ncols();
        if X.cols() != n_features {
            return Err(format!(
                "Feature mismatch: expected {}, got {}",
                n_features,
                X.cols()
            ));
        }

        if classes.len() == 2 {
            let mut preds = Vec::with_capacity(X.rows());
            for vec in X.outer_iterator() {
                let mut z = intercept[0];
                for (col_idx, &val) in vec.iter() {
                    z += val * coef[[0, col_idx]];
                }
                preds.push(if Self::sigmoid(z) >= 0.5 { classes[1] } else { classes[0] });
            }
            Ok(Array1::from(preds))
        } else {
            let n_classes = classes.len();
            let mut preds = Vec::with_capacity(X.rows());
            for vec in X.outer_iterator() {
                let mut best_score = f64::NEG_INFINITY;
                let mut best_class = classes[0];
                for c in 0..n_classes {
                    let mut z = intercept[c];
                    for (col_idx, &val) in vec.iter() {
                        z += val * coef[[c, col_idx]];
                    }
                    if z > best_score {
                        best_score = z;
                        best_class = classes[c];
                    }
                }
                preds.push(best_class);
            }
            Ok(Array1::from(preds))
        }
    }

    /// Predict probabilities for sparse matrix.
    pub fn predict_proba_sparse(&self, X: &CsMat<f64>) -> Result<Array2<f64>, String> {
        let coef = self.coef_.as_ref().ok_or("Model is not fitted yet")?;
        let intercept = self.intercept_.as_ref().unwrap();
        let classes = self.classes_.as_ref().unwrap();

        let n_features = coef.ncols();
        if X.cols() != n_features {
            return Err(format!(
                "Feature mismatch: expected {}, got {}",
                n_features,
                X.cols()
            ));
        }

        let n = X.rows();

        if classes.len() == 2 {
            let mut proba = Array2::<f64>::zeros((n, 2));
            for (i, vec) in X.outer_iterator().enumerate() {
                let mut z = intercept[0];
                for (col_idx, &val) in vec.iter() {
                    z += val * coef[[0, col_idx]];
                }
                let p = Self::sigmoid(z);
                proba[[i, 0]] = 1.0 - p;
                proba[[i, 1]] = p;
            }
            Ok(proba)
        } else {
            let n_classes = classes.len();
            let mut proba = Array2::<f64>::zeros((n, n_classes));
            for (i, vec) in X.outer_iterator().enumerate() {
                let mut row_sum = 0.0;
                for c in 0..n_classes {
                    let mut z = intercept[c];
                    for (col_idx, &val) in vec.iter() {
                        z += val * coef[[c, col_idx]];
                    }
                    let p = Self::sigmoid(z);
                    proba[[i, c]] = p;
                    row_sum += p;
                }
                if row_sum > 0.0 {
                    for c in 0..n_classes {
                        proba[[i, c]] /= row_sum;
                    }
                }
            }
            Ok(proba)
        }
    }

    /// Accuracy score.
    pub fn score(&self, X: &ArrayView2<f64>, y: &ArrayView1<f64>) -> Result<f64, String> {
        let preds = self.predict(X)?;
        metrics::accuracy_score(y.as_slice().unwrap(), preds.as_slice().unwrap())
    }
}

// ==========================================
// Tests
// ==========================================
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    // ---- LinearRegression tests ----

    #[test]
    fn test_linear_regression_simple() {
        // y = 2*x + 1
        let X = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let y = array![3.0, 5.0, 7.0, 9.0, 11.0];
        let mut lr = LinearRegression::new(true);
        lr.fit(&X.view(), &y.view()).unwrap();

        let coef = lr.coef_.as_ref().unwrap();
        assert!((coef[0] - 2.0).abs() < 1e-6, "coef should be ~2.0, got {}", coef[0]);
        assert!((lr.intercept_ - 1.0).abs() < 1e-6, "intercept should be ~1.0, got {}", lr.intercept_);

        let preds = lr.predict(&X.view()).unwrap();
        for i in 0..5 {
            assert!((preds[i] - y[i]).abs() < 1e-6);
        }

        let r2 = lr.score(&X.view(), &y.view()).unwrap();
        assert!((r2 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_linear_regression_multivariate() {
        // y = 1*x0 + 2*x1 + 3
        let X = array![
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [1.0, 2.0]
        ];
        let y = array![4.0, 5.0, 6.0, 7.0, 8.0];
        let mut lr = LinearRegression::new(true);
        lr.fit(&X.view(), &y.view()).unwrap();

        let coef = lr.coef_.as_ref().unwrap();
        // Expected coefs are [1.0, 2.0] and intercept 3.0
        assert!((coef[0] - 1.0).abs() < 1e-6);
        assert!((coef[1] - 2.0).abs() < 1e-6);
        assert!((lr.intercept_ - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_linear_regression_no_intercept() {
        // y = 3*x
        let X = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![3.0, 6.0, 9.0, 12.0];
        let mut lr = LinearRegression::new(false);
        lr.fit(&X.view(), &y.view()).unwrap();

        let coef = lr.coef_.as_ref().unwrap();
        assert!((coef[0] - 3.0).abs() < 1e-6);
        assert!((lr.intercept_).abs() < 1e-12);
    }

    // ---- Ridge tests ----

    #[test]
    fn test_ridge_simple() {
        let X = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let y = array![3.0, 5.0, 7.0, 9.0, 11.0];
        let mut ridge = Ridge::new(0.0, true);
        ridge.fit(&X.view(), &y.view()).unwrap();

        // With alpha=0, should be same as OLS
        let coef = ridge.coef_.as_ref().unwrap();
        assert!((coef[0] - 2.0).abs() < 1e-6);
        assert!((ridge.intercept_ - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_ridge_regularization() {
        // With high alpha, coefficients should shrink toward 0
        let X = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let y = array![3.0, 5.0, 7.0, 9.0, 11.0];

        let mut ridge_low = Ridge::new(0.01, true);
        ridge_low.fit(&X.view(), &y.view()).unwrap();

        let mut ridge_high = Ridge::new(100.0, true);
        ridge_high.fit(&X.view(), &y.view()).unwrap();

        // Higher alpha => smaller coefficient magnitude
        let coef_low = ridge_low.coef_.as_ref().unwrap()[0].abs();
        let coef_high = ridge_high.coef_.as_ref().unwrap()[0].abs();
        assert!(coef_high < coef_low, "Higher alpha should shrink coefficients");
    }

    // ---- Lasso tests ----

    #[test]
    fn test_lasso_simple() {
        let X = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let y = array![3.0, 5.0, 7.0, 9.0, 11.0];
        let mut lasso = Lasso::new(0.0001, true, 10000, 1e-8);
        lasso.fit(&X.view(), &y.view()).unwrap();

        let coef = lasso.coef_.as_ref().unwrap();
        assert!((coef[0] - 2.0).abs() < 0.05, "coef should be ~2.0, got {}", coef[0]);
    }

    #[test]
    fn test_lasso_sparsity() {
        // High alpha should drive coefficients to zero
        let X = array![
            [1.0, 0.1],
            [2.0, 0.2],
            [3.0, 0.3],
            [4.0, 0.4],
            [5.0, 0.5]
        ];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0]; // y  2*x0
        let mut lasso = Lasso::new(5.0, true, 10000, 1e-8);
        lasso.fit(&X.view(), &y.view()).unwrap();

        let coef = lasso.coef_.as_ref().unwrap();
        // With high alpha, some coefficients should be exactly 0
        let n_zero = coef.iter().filter(|&&c| c.abs() < 1e-10).count();
        assert!(n_zero >= 1, "Lasso should produce sparse coefficients");
    }

    // ---- LogisticRegression tests ----

    #[test]
    fn test_logistic_regression_binary() {
        // Simple linearly separable data
        let X = array![
            [-1.0, -1.0],
            [-0.5, -0.5],
            [-0.2, -0.2],
            [0.5, 0.5],
            [0.8, 0.8],
            [1.0, 1.0]
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let mut lr = LogisticRegression::new(1.0, 1000, 1e-6, "l2");
        lr.fit(&X.view(), &y.view()).unwrap();

        let preds = lr.predict(&X.view()).unwrap();
        // Should classify correctly
        for i in 0..3 {
            assert!((preds[i] - 0.0).abs() < f64::EPSILON, "Sample {} should be class 0", i);
        }
        for i in 3..6 {
            assert!((preds[i] - 1.0).abs() < f64::EPSILON, "Sample {} should be class 1", i);
        }

        let acc = lr.score(&X.view(), &y.view()).unwrap();
        assert!((acc - 1.0).abs() < 1e-9, "Should achieve perfect accuracy on linearly separable data");
    }

    #[test]
    fn test_logistic_regression_predict_proba() {
        let X = array![
            [0.0, 0.0],
            [1.0, 1.0]
        ];
        let y = array![0.0, 1.0];

        let mut lr = LogisticRegression::new(1.0, 1000, 1e-6, "l2");
        lr.fit(&X.view(), &y.view()).unwrap();

        let proba = lr.predict_proba(&X.view()).unwrap();
        // Probabilities should sum to 1 for each row
        for i in 0..2 {
            let row_sum = proba[[i, 0]] + proba[[i, 1]];
            assert!((row_sum - 1.0).abs() < 1e-9, "Probabilities should sum to 1");
        }
        // First sample should have higher prob for class 0
        assert!(proba[[0, 0]] > proba[[0, 1]]);
        // Second sample should have higher prob for class 1
        assert!(proba[[1, 1]] > proba[[1, 0]]);
    }

    #[test]
    fn test_logistic_regression_multiclass() {
        // Three linearly separable clusters
        let X = array![
            [0.0, 0.0],
            [0.1, 0.0],
            [5.0, 0.0],
            [5.1, 0.0],
            [0.0, 5.0],
            [0.0, 5.1]
        ];
        let y = array![0.0, 0.0, 1.0, 1.0, 2.0, 2.0];

        let mut lr = LogisticRegression::new(1.0, 2000, 1e-6, "l2");
        lr.fit(&X.view(), &y.view()).unwrap();

        let preds = lr.predict(&X.view()).unwrap();
        let acc = lr.score(&X.view(), &y.view()).unwrap();
        assert!(acc >= 0.8, "Multiclass accuracy should be reasonable, got {}", acc);

        let classes = lr.classes_.as_ref().unwrap();
        assert_eq!(classes.len(), 3);
    }

    // ---- solve_linear_system tests ----

    #[test]
    fn test_solve_identity() {
        let A = array![[1.0, 0.0], [0.0, 1.0]];
        let b = array![[3.0], [5.0]];
        let x = solve_linear_system(&A, &b).unwrap();
        assert!((x[[0, 0]] - 3.0).abs() < 1e-12);
        assert!((x[[1, 0]] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_solve_2x2() {
        // [2 1; 5 3] * x = [11; 29]  =>  x = [4; 3]
        let A = array![[2.0, 1.0], [5.0, 3.0]];
        let b = array![[11.0], [29.0]];
        let x = solve_linear_system(&A, &b).unwrap();
        assert!((x[[0, 0]] - 4.0).abs() < 1e-10);
        assert!((x[[1, 0]] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_unfitted_predict_errors() {
        let X = array![[1.0, 2.0]];
        let lr = LinearRegression::new(true);
        assert!(lr.predict(&X.view()).is_err());

        let ridge = Ridge::new(1.0, true);
        assert!(ridge.predict(&X.view()).is_err());

        let lasso = Lasso::new(1.0, true, 100, 1e-4);
        assert!(lasso.predict(&X.view()).is_err());

        let logreg = LogisticRegression::new(1.0, 100, 1e-4, "l2");
        assert!(logreg.predict(&X.view()).is_err());
    }
}

// ==========================================
// LinearSVC  Linear Support Vector Classification
// ==========================================
pub struct LinearSVC {
    pub coef_: Option<Array2<f64>>,
    pub intercept_: Option<Array1<f64>>,
    pub classes_: Option<Vec<f64>>,
    pub C: f64,
    pub max_iter: usize,
    pub tol: f64,
    pub learning_rate: f64,
}

impl LinearSVC {
    pub fn new(C: f64, max_iter: usize, tol: f64) -> Self {
        LinearSVC {
            coef_: None,
            intercept_: None,
            classes_: None,
            C,
            max_iter,
            tol,
            learning_rate: 0.01,
        }
    }

    fn fit_binary(
        &self,
        X: &Array2<f64>,
        y_binary: &Array1<f64>, // -1 or 1 labels
        n_features: usize,
    ) -> Result<(Array1<f64>, f64), String> {
        let n_f64 = X.nrows() as f64;
        let lambda = 1.0 / self.C;

        let mut w = Array1::<f64>::zeros(n_features);
        let mut bias = 0.0_f64;
        let lr = self.learning_rate;

        for _iter in 0..self.max_iter {
            let z = X.dot(&w) + bias;
            // Hinge loss gradient: if y*z < 1, grad_w += -y*X, grad_b += -y
            let mut grad_w = &w * lambda;
            let mut grad_b = 0.0;
            
            for i in 0..X.nrows() {
                if y_binary[i] * z[i] < 1.0 {
                    let update = -y_binary[i] / n_f64;
                    for j in 0..n_features {
                        grad_w[j] += update * X[[i, j]];
                    }
                    grad_b += update;
                }
            }

            let w_update = &grad_w * lr;
            w = w - &w_update;
            let b_update = grad_b * lr;
            bias -= b_update;

            let max_change = w_update.iter().map(|v| v.abs()).fold(0.0_f64, |a, b| a.max(b)).max(b_update.abs());
            if max_change < self.tol {
                break;
            }
        }

        Ok((w, bias))
    }

    fn fit_binary_sparse(
        &self,
        X: &CsMat<f64>,
        y_binary: &Array1<f64>, // -1 or 1 labels
        n_features: usize,
    ) -> Result<(Array1<f64>, f64), String> {
        let n_f64 = X.rows() as f64;
        let lambda = 1.0 / self.C;

        let mut w = Array1::<f64>::zeros(n_features);
        let mut bias = 0.0_f64;
        let lr = self.learning_rate;

        for _iter in 0..self.max_iter {
            let mut grad_w = &w * lambda;
            let mut grad_b = 0.0;
            
            for (i, vec) in X.outer_iterator().enumerate() {
                let mut z = bias;
                for (col_idx, &val) in vec.iter() {
                    z += val * w[col_idx];
                }
                
                if y_binary[i] * z < 1.0 {
                    let update = -y_binary[i] / n_f64;
                    for (col_idx, &val) in vec.iter() {
                        grad_w[col_idx] += update * val;
                    }
                    grad_b += update;
                }
            }

            let w_update = &grad_w * lr;
            w = w - &w_update;
            let b_update = grad_b * lr;
            bias -= b_update;

            let max_change = w_update.iter().map(|v| v.abs()).fold(0.0_f64, |a, b| a.max(b)).max(b_update.abs());
            if max_change < self.tol {
                break;
            }
        }

        Ok((w, bias))
    }

    pub fn fit(&mut self, X: &ArrayView2<f64>, y: &ArrayView1<f64>) -> Result<(), String> {
        if X.is_empty() { return Err("Input array is empty".to_string()); }
        check_finite_2d(X)?;
        check_finite_1d(y)?;

        let mut unique_classes = y.to_vec();
        unique_classes.sort_by(|a, b| a.partial_cmp(b).unwrap());
        unique_classes.dedup();
        let classes = unique_classes;

        if classes.len() < 2 {
            return Err("LinearSVC requires at least 2 classes".to_string());
        }

        let n_features = X.ncols();

        if classes.len() == 2 {
            let mut y_bin = Array1::<f64>::zeros(y.len());
            for i in 0..y.len() {
                y_bin[i] = if y[i] == classes[1] { 1.0 } else { -1.0 };
            }
            let (w, b) = self.fit_binary(&X.to_owned(), &y_bin, n_features)?;
            
            let intercept = Array1::from(vec![b]);
            let mut coef = Array2::<f64>::zeros((1, n_features));
            for j in 0..n_features {
                coef[[0, j]] = w[j];
            }

            self.coef_ = Some(coef);
            self.intercept_ = Some(intercept);
        } else {
            // Multiclass: one-vs-rest
            let n_classes = classes.len();
            let mut coef = Array2::<f64>::zeros((n_classes, n_features));
            let mut intercept = Array1::<f64>::zeros(n_classes);

            for (c_idx, &cls) in classes.iter().enumerate() {
                let mut y_bin = Array1::<f64>::zeros(y.len());
                for i in 0..y.len() {
                    y_bin[i] = if (y[i] - cls).abs() < f64::EPSILON { 1.0 } else { -1.0 };
                }
                let (w, b) = self.fit_binary(&X.to_owned(), &y_bin, n_features)?;
                for j in 0..n_features {
                    coef[[c_idx, j]] = w[j];
                }
                intercept[c_idx] = b;
            }
            self.coef_ = Some(coef);
            self.intercept_ = Some(intercept);
        }

        self.classes_ = Some(classes);
        Ok(())
    }

    pub fn fit_sparse(&mut self, X: &CsMat<f64>, y: &ArrayView1<f64>) -> Result<(), String> {
        if X.rows() == 0 || X.cols() == 0 { return Err("Input array is empty".to_string()); }
        check_finite_1d(y)?;

        let mut unique_classes = y.to_vec();
        unique_classes.sort_by(|a, b| a.partial_cmp(b).unwrap());
        unique_classes.dedup();
        let classes = unique_classes;

        if classes.len() < 2 {
            return Err("LinearSVC requires at least 2 classes".to_string());
        }

        let n_features = X.cols();

        if classes.len() == 2 {
            let mut y_bin = Array1::<f64>::zeros(y.len());
            for i in 0..y.len() {
                y_bin[i] = if (y[i] - classes[1]).abs() < f64::EPSILON { 1.0 } else { -1.0 };
            }
            let (w, b) = self.fit_binary_sparse(X, &y_bin, n_features)?;
            
            let intercept = Array1::from(vec![b]);
            let mut coef = Array2::<f64>::zeros((1, n_features));
            for j in 0..n_features {
                coef[[0, j]] = w[j];
            }

            self.coef_ = Some(coef);
            self.intercept_ = Some(intercept);
        } else {
            // Multiclass: one-vs-rest
            let n_classes = classes.len();
            let mut coef = Array2::<f64>::zeros((n_classes, n_features));
            let mut intercept = Array1::<f64>::zeros(n_classes);

            for (c_idx, &cls) in classes.iter().enumerate() {
                let mut y_bin = Array1::<f64>::zeros(y.len());
                for i in 0..y.len() {
                    y_bin[i] = if (y[i] - cls).abs() < f64::EPSILON { 1.0 } else { -1.0 };
                }
                let (w, b) = self.fit_binary_sparse(X, &y_bin, n_features)?;
                for j in 0..n_features {
                    coef[[c_idx, j]] = w[j];
                }
                intercept[c_idx] = b;
            }
            self.coef_ = Some(coef);
            self.intercept_ = Some(intercept);
        }

        self.classes_ = Some(classes);
        Ok(())
    }

    pub fn predict(&self, X: &ArrayView2<f64>) -> Result<Array1<f64>, String> {
        let coef = self.coef_.as_ref().ok_or("Model is not fitted yet")?;
        let intercept = self.intercept_.as_ref().unwrap();
        let classes = self.classes_.as_ref().unwrap();

        let n_samples = X.nrows();
        let mut preds = Array1::<f64>::zeros(n_samples);

        if classes.len() == 2 {
            for i in 0..n_samples {
                let mut z = intercept[0];
                for j in 0..X.ncols() {
                    z += X[[i, j]] * coef[[0, j]];
                }
                preds[i] = if z >= 0.0 { classes[1] } else { classes[0] };
            }
        } else {
            let n_classes = classes.len();
            for i in 0..n_samples {
                let mut best_class = 0;
                let mut max_z = f64::NEG_INFINITY;
                for c in 0..n_classes {
                    let mut z = intercept[c];
                    for j in 0..X.ncols() {
                        z += X[[i, j]] * coef[[c, j]];
                    }
                    if z > max_z {
                        max_z = z;
                        best_class = c;
                    }
                }
                preds[i] = classes[best_class];
            }
        }

        Ok(Array1::from(preds))
    }

    pub fn predict_sparse(&self, X: &CsMat<f64>) -> Result<Array1<f64>, String> {
        let coef = self.coef_.as_ref().ok_or("Model is not fitted yet")?;
        let intercept = self.intercept_.as_ref().unwrap();
        let classes = self.classes_.as_ref().unwrap();

        let n_samples = X.rows();
        let mut preds = Array1::<f64>::zeros(n_samples);

        if classes.len() == 2 {
            for (i, vec) in X.outer_iterator().enumerate() {
                let mut z = intercept[0];
                for (col_idx, &val) in vec.iter() {
                    z += val * coef[[0, col_idx]];
                }
                preds[i] = if z >= 0.0 { classes[1] } else { classes[0] };
            }
        } else {
            let n_classes = classes.len();
            for (i, vec) in X.outer_iterator().enumerate() {
                let mut best_class = 0;
                let mut max_z = f64::NEG_INFINITY;
                for c in 0..n_classes {
                    let mut z = intercept[c];
                    for (col_idx, &val) in vec.iter() {
                        z += val * coef[[c, col_idx]];
                    }
                    if z > max_z {
                        max_z = z;
                        best_class = c;
                    }
                }
                preds[i] = classes[best_class];
            }
        }

        Ok(Array1::from(preds))
    }
}
