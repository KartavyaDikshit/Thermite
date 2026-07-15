use ndarray::{Array1, Array2, ArrayView2};
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone)]
pub struct ALS {
    pub factors: usize,
    pub iterations: usize,
    pub regularization: f64,
    pub user_factors: Option<Array2<f64>>,
    pub item_factors: Option<Array2<f64>>,
}

impl ALS {
    pub fn new(factors: usize, iterations: usize, regularization: f64) -> Self {
        ALS {
            factors,
            iterations,
            regularization,
            user_factors: None,
            item_factors: None,
        }
    }

    pub fn fit(&mut self, R: &ArrayView2<f64>) -> Result<(), String> {
        let n_users = R.nrows();
        let n_items = R.ncols();
        let mut rng = rand::thread_rng();

        let mut user_factors = Array2::from_shape_fn((n_users, self.factors), |_| rng.gen_range(0.0..1.0));
        let mut item_factors = Array2::from_shape_fn((n_items, self.factors), |_| rng.gen_range(0.0..1.0));

        let reg = self.regularization;

        for _ in 0..self.iterations {
            // Update user factors: for each user, solve (V^T V + reg*I) * u_i = V^T * r_i
            for i in 0..n_users {
                let mut A = Array2::<f64>::zeros((self.factors, self.factors));
                let mut b = Array1::<f64>::zeros(self.factors);
                for j in 0..n_items {
                    if R[[i, j]] > 0.0 {
                        for f1 in 0..self.factors {
                            for f2 in 0..self.factors {
                                A[[f1, f2]] += item_factors[[j, f1]] * item_factors[[j, f2]];
                            }
                            b[f1] += item_factors[[j, f1]] * R[[i, j]];
                        }
                    }
                }
                for f in 0..self.factors {
                    A[[f, f]] += reg;
                }
                // Solve A * u = b via Cholesky-like (simple Gaussian elimination for small factors)
                let nf = self.factors;
                let mut A_copy = A.clone();
                let mut b_copy = b.clone();
                for col in 0..nf {
                    let pivot = A_copy[[col, col]];
                    if pivot.abs() < 1e-12 { continue; }
                    for row in (col + 1)..nf {
                        let factor = A_copy[[row, col]] / pivot;
                        for k in col..nf {
                            let idx = k;
                            A_copy[[row, idx]] -= factor * A_copy[[col, idx]];
                        }
                        b_copy[row] -= factor * b_copy[col];
                    }
                }
                for col in (0..nf).rev() {
                    let pivot = A_copy[[col, col]];
                    if pivot.abs() > 1e-12 {
                        let mut sum = b_copy[col];
                        for k in (col + 1)..nf {
                            sum -= A_copy[[col, k]] * user_factors[[i, k]];
                        }
                        user_factors[[i, col]] = sum / pivot;
                    }
                }
            }

            // Update item factors: for each item, solve (U^T U + reg*I) * v_j = U^T * r_j
            for j in 0..n_items {
                let mut A = Array2::<f64>::zeros((self.factors, self.factors));
                let mut b = Array1::<f64>::zeros(self.factors);
                for i in 0..n_users {
                    if R[[i, j]] > 0.0 {
                        for f1 in 0..self.factors {
                            for f2 in 0..self.factors {
                                A[[f1, f2]] += user_factors[[i, f1]] * user_factors[[i, f2]];
                            }
                            b[f1] += user_factors[[i, f1]] * R[[i, j]];
                        }
                    }
                }
                for f in 0..self.factors {
                    A[[f, f]] += reg;
                }
                // Solve A * v = b via Gaussian elimination
                let nf = self.factors;
                let mut A_copy = A.clone();
                let mut b_copy = b.clone();
                for col in 0..nf {
                    let pivot = A_copy[[col, col]];
                    if pivot.abs() < 1e-12 { continue; }
                    for row in (col + 1)..nf {
                        let factor = A_copy[[row, col]] / pivot;
                        for k in col..nf {
                            A_copy[[row, k]] -= factor * A_copy[[col, k]];
                        }
                        b_copy[row] -= factor * b_copy[col];
                    }
                }
                for col in (0..nf).rev() {
                    let pivot = A_copy[[col, col]];
                    if pivot.abs() > 1e-12 {
                        let mut sum = b_copy[col];
                        for k in (col + 1)..nf {
                            sum -= A_copy[[col, k]] * item_factors[[j, k]];
                        }
                        item_factors[[j, col]] = sum / pivot;
                    }
                }
            }
        }

        self.user_factors = Some(user_factors);
        self.item_factors = Some(item_factors);

        Ok(())
    }

    pub fn predict(&self, user_id: usize, item_id: usize) -> Result<f64, String> {
        if let (Some(u_f), Some(i_f)) = (&self.user_factors, &self.item_factors) {
            let mut score = 0.0;
            for f in 0..self.factors {
                score += u_f[[user_id, f]] * i_f[[item_id, f]];
            }
            Ok(score)
        } else {
            Err("Model not fitted".to_string())
        }
    }
}
