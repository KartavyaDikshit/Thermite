#![allow(non_snake_case)]
use ndarray::{Array1, Array2, ArrayView2};
use rand::Rng;

fn multivariate_gaussian_log_prob(x: &[f64], mean: &[f64], cov: &Array2<f64>) -> f64 {
    let k = x.len();
    let mut diff = Array1::from_shape_fn(k, |i| x[i] - mean[i]);
    // Solve cov * sol = diff for sol (linear system)
    let mut cov_copy = cov.to_owned();
    let mut rhs = diff.clone();
    let n = k;
    for col in 0..n {
        let pivot = cov_copy[[col, col]];
        if pivot.abs() < 1e-12 { continue; }
        for row in (col + 1)..n {
            let factor = cov_copy[[row, col]] / pivot;
            for c in col..n {
                cov_copy[[row, c]] -= factor * cov_copy[[col, c]];
            }
            rhs[row] -= factor * rhs[col];
        }
    }
    for col in (0..n).rev() {
        let pivot = cov_copy[[col, col]];
        if pivot.abs() > 1e-12 {
            let mut sum = rhs[col];
            for k in (col + 1)..n {
                sum -= cov_copy[[col, k]] * diff[k];
            }
            diff[col] = sum / pivot;
        }
    }

    let mut log_det = 0.0;
    for i in 0..n {
        log_det += cov_copy[[i, i]].abs().ln();
    }

    let quad_form = diff.dot(&diff);
    -0.5 * (n as f64 * (2.0 * std::f64::consts::PI).ln() + log_det + quad_form)
}

pub struct GaussianMixture {
    pub n_components: usize,
    pub max_iter: usize,
    pub tol: f64,
    pub means_: Option<Array2<f64>>,
    pub weights_: Option<Array1<f64>>,
    pub covariances_: Option<Vec<Array2<f64>>>,
}

impl GaussianMixture {
    pub fn new(n_components: usize, max_iter: usize, tol: f64) -> Self {
        GaussianMixture {
            n_components,
            max_iter,
            tol,
            means_: None,
            weights_: None,
            covariances_: None,
        }
    }

    pub fn fit(&mut self, X: &ArrayView2<f64>) -> Result<(), String> {
        let n_samples = X.nrows();
        let n_features = X.ncols();
        if n_samples < self.n_components {
            return Err("n_samples < n_components".to_string());
        }

        // Initialize means with k-means++ style
        let mut rng = rand::thread_rng();
        let mut means = Array2::<f64>::zeros((self.n_components, n_features));
        let first_idx = rng.gen_range(0..n_samples);
        for j in 0..n_features {
            means[[0, j]] = X[[first_idx, j]];
        }
        for k in 1..self.n_components {
            let mut dists = Array1::<f64>::zeros(n_samples);
            for i in 0..n_samples {
                let mut min_d = f64::MAX;
                for k2 in 0..k {
                    let mut d = 0.0;
                    for j in 0..n_features {
                        let diff = X[[i, j]] - means[[k2, j]];
                        d += diff * diff;
                    }
                    if d < min_d { min_d = d; }
                }
                dists[i] = min_d;
            }
            let total: f64 = dists.iter().sum();
            if total < 1e-12 { break; }
            let mut r = rng.gen::<f64>() * total;
            let mut idx = 0;
            for i in 0..n_samples {
                r -= dists[i];
                if r <= 0.0 { idx = i; break; }
            }
            for j in 0..n_features {
                means[[k, j]] = X[[idx, j]];
            }
        }

        let mut weights = Array1::<f64>::from_elem(self.n_components, 1.0 / (self.n_components as f64));
        let mut covariances = Vec::with_capacity(self.n_components);
        for _ in 0..self.n_components {
            let cov = Array2::<f64>::eye(n_features);
            covariances.push(cov);
        }

        // EM algorithm
        let mut prev_log_lik = f64::NEG_INFINITY;
        for _iter in 0..self.max_iter {
            // E-step: compute responsibilities
            let mut resp = Array2::<f64>::zeros((n_samples, self.n_components));
            for i in 0..n_samples {
                let x_row: Vec<f64> = X.row(i).iter().copied().collect();
                let mut log_probs = Array1::<f64>::zeros(self.n_components);
                for k in 0..self.n_components {
                    let mean: Vec<f64> = means.row(k).iter().copied().collect();
                    log_probs[k] = multivariate_gaussian_log_prob(&x_row, &mean, &covariances[k]);
                }
                let max_log = log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let mut sum_exp = 0.0;
                for k in 0..self.n_components {
                    resp[[i, k]] = (log_probs[k] - max_log).exp();
                    sum_exp += resp[[i, k]];
                }
                if sum_exp > 0.0 {
                    for k in 0..self.n_components {
                        resp[[i, k]] /= sum_exp;
                    }
                }
            }

            // M-step: update means, covariances, weights
            let n_soft: Array1<f64> = Array1::from_shape_fn(self.n_components, |k| {
                (0..n_samples).map(|i| resp[[i, k]]).sum()
            });
            let total = n_soft.sum();
            if total > 0.0 {
                for k in 0..self.n_components {
                    weights[k] = n_soft[k] / total;
                }
            }

            for k in 0..self.n_components {
                if n_soft[k] < 1e-12 { continue; }
                for j in 0..n_features {
                    let mut s = 0.0;
                    for i in 0..n_samples {
                        s += resp[[i, k]] * X[[i, j]];
                    }
                    means[[k, j]] = s / n_soft[k];
                }
            }

            for k in 0..self.n_components {
                if n_soft[k] < 1e-12 { continue; }
                let mut cov = Array2::<f64>::zeros((n_features, n_features));
                for i in 0..n_samples {
                    let r = resp[[i, k]];
                    if r < 1e-12 { continue; }
                    for j1 in 0..n_features {
                        for j2 in 0..n_features {
                            cov[[j1, j2]] += r * (X[[i, j1]] - means[[k, j1]]) * (X[[i, j2]] - means[[k, j2]]);
                        }
                    }
                }
                for j in 0..n_features {
                    cov[[j, j]] += 1e-6; // regularization
                }
                cov /= n_soft[k];
                covariances[k] = cov;
            }

            // Check log-likelihood convergence
            let mut log_lik = 0.0;
            for i in 0..n_samples {
                let x_row: Vec<f64> = X.row(i).iter().copied().collect();
                let mut max_log = f64::NEG_INFINITY;
                for k in 0..self.n_components {
                    let mean: Vec<f64> = means.row(k).iter().copied().collect();
                    let lp = multivariate_gaussian_log_prob(&x_row, &mean, &covariances[k]);
                    let w = weights[k].ln();
                    let val = w + lp;
                    if val > max_log { max_log = val; }
                }
                let mut sum_exp = 0.0;
                for k in 0..self.n_components {
                    let mean: Vec<f64> = means.row(k).iter().copied().collect();
                    let lp = multivariate_gaussian_log_prob(&x_row, &mean, &covariances[k]);
                    sum_exp += (weights[k].ln() + lp - max_log).exp();
                }
                log_lik += max_log + sum_exp.ln();
            }

            if _iter > 0 && (log_lik - prev_log_lik).abs() < self.tol {
                break;
            }
            prev_log_lik = log_lik;
        }

        self.means_ = Some(means);
        self.weights_ = Some(weights);
        self.covariances_ = Some(covariances);

        Ok(())
    }

    pub fn predict(&self, X: &ArrayView2<f64>) -> Result<Vec<usize>, String> {
        if self.means_.is_none() {
            return Err("Model not fitted".to_string());
        }
        let means = self.means_.as_ref().unwrap();
        let covariances = self.covariances_.as_ref().unwrap();
        let n_samples = X.nrows();
        let mut predictions = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            let x_row: Vec<f64> = X.row(i).iter().copied().collect();
            let mut best_k = 0;
            let mut best_log_prob = f64::NEG_INFINITY;
            for k in 0..self.n_components {
                let mean: Vec<f64> = means.row(k).iter().copied().collect();
                let lp = multivariate_gaussian_log_prob(&x_row, &mean, &covariances[k]);
                if lp > best_log_prob {
                    best_log_prob = lp;
                    best_k = k;
                }
            }
            predictions.push(best_k);
        }
        Ok(predictions)
    }

    pub fn predict_proba(&self, X: &ArrayView2<f64>) -> Result<Array2<f64>, String> {
        if self.means_.is_none() {
            return Err("Model not fitted".to_string());
        }
        let means = self.means_.as_ref().unwrap();
        let covariances = self.covariances_.as_ref().unwrap();
        let weights = self.weights_.as_ref().unwrap();
        let n_samples = X.nrows();
        let mut proba = Array2::<f64>::zeros((n_samples, self.n_components));
        for i in 0..n_samples {
            let x_row: Vec<f64> = X.row(i).iter().copied().collect();
            let mut max_log = f64::NEG_INFINITY;
            let mut log_probs = Array1::<f64>::zeros(self.n_components);
            for k in 0..self.n_components {
                let mean: Vec<f64> = means.row(k).iter().copied().collect();
                let lp = multivariate_gaussian_log_prob(&x_row, &mean, &covariances[k]);
                log_probs[k] = weights[k].ln() + lp;
                if log_probs[k] > max_log { max_log = log_probs[k]; }
            }
            let mut sum_exp = 0.0;
            for k in 0..self.n_components {
                proba[[i, k]] = (log_probs[k] - max_log).exp();
                sum_exp += proba[[i, k]];
            }
            if sum_exp > 0.0 {
                for k in 0..self.n_components {
                    proba[[i, k]] /= sum_exp;
                }
            }
        }
        Ok(proba)
    }
}
