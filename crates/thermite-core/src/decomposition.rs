#![allow(non_snake_case)]
use ndarray::{Array1, Array2, ArrayView2, Axis};
use rand::prelude::*;
use rand::rngs::SmallRng;
use rayon::prelude::*;

fn check_finite(X: &ArrayView2<f64>) -> Result<(), String> {
    if X.iter().any(|&val| !val.is_finite()) {
        return Err("Input contains NaN or infinity values".to_string());
    }
    Ok(())
}

// ==========================================
// PCA (Principal Component Analysis)
// ==========================================
/// PCA via eigendecomposition of the covariance matrix.
///
/// Uses the power iteration / deflation method to extract
/// the top-k eigenvectors of X^T X (covariance matrix).
pub struct PCA {
    pub n_components: usize,
    pub random_state: Option<u64>,
    // Fitted attributes
    pub components_: Option<Array2<f64>>,
    pub explained_variance_: Option<Array1<f64>>,
    pub explained_variance_ratio_: Option<Array1<f64>>,
    pub mean_: Option<Array1<f64>>,
}

impl PCA {
    pub fn new(n_components: usize, random_state: Option<u64>) -> Self {
        PCA {
            n_components,
            random_state,
            components_: None,
            explained_variance_: None,
            explained_variance_ratio_: None,
            mean_: None,
        }
    }

    /// Compute column-wise mean (parallel over columns).
    fn compute_mean(X: &ArrayView2<f64>) -> Array1<f64> {
        let n_samples = X.nrows() as f64;
        let means: Vec<f64> = X
            .axis_iter(Axis(1))
            .into_par_iter()
            .map(|col| col.sum() / n_samples)
            .collect();
        Array1::from(means)
    }

    /// Center X by subtracting the mean row-wise.
    fn center(X: &ArrayView2<f64>, mean: &Array1<f64>) -> Array2<f64> {
        let mut X_centered = X.to_owned();
        X_centered
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .for_each(|mut row| {
                for j in 0..row.len() {
                    row[j] -= mean[j];
                }
            });
        X_centered
    }

    /// Power iteration to find the dominant eigenvector of a symmetric matrix A.
    /// Returns (eigenvalue, eigenvector).
    fn power_iteration(
        A: &Array2<f64>,
        rng: &mut SmallRng,
        max_iter: usize,
        tol: f64,
    ) -> (f64, Array1<f64>) {
        let n = A.nrows();
        let mut v = Array1::zeros(n);
        for i in 0..n {
            v[i] = rng.gen::<f64>() - 0.5;
        }
        // Normalize
        let norm = v.dot(&v).sqrt();
        if norm > 0.0 {
            v.mapv_inplace(|x| x / norm);
        }

        let mut eigenvalue = 0.0;

        for _ in 0..max_iter {
            // w = A * v
            let w = A.dot(&v);
            let new_eigenvalue = v.dot(&w);
            let w_norm = w.dot(&w).sqrt();

            if w_norm < 1e-15 {
                break;
            }

            let new_v = &w / w_norm;

            // Check convergence
            let diff: f64 = new_v
                .iter()
                .zip(v.iter())
                .map(|(&a, &b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();

            v = new_v;
            eigenvalue = new_eigenvalue;

            if diff < tol {
                break;
            }
        }

        (eigenvalue, v)
    }

    /// Deflate matrix: A' = A - eigenvalue * v * v^T
    fn deflate(A: &mut Array2<f64>, eigenvalue: f64, v: &Array1<f64>) {
        let n = A.nrows();
        for i in 0..n {
            for j in 0..n {
                A[[i, j]] -= eigenvalue * v[i] * v[j];
            }
        }
    }

    pub fn fit(&mut self, X: &ArrayView2<f64>) -> Result<(), String> {
        if X.is_empty() {
            return Err("Input array is empty".to_string());
        }
        check_finite(X)?;

        let n_samples = X.nrows();
        let n_features = X.ncols();

        if self.n_components > n_features {
            return Err(format!(
                "n_components={} must be <= n_features={}",
                self.n_components, n_features
            ));
        }

        let mean = Self::compute_mean(X);
        let X_centered = Self::center(X, &mean);

        // Covariance matrix: (1 / (n-1)) * X_c^T * X_c
        let denom = if n_samples > 1 {
            (n_samples - 1) as f64
        } else {
            1.0
        };

        let mut cov = X_centered.t().dot(&X_centered);
        cov.mapv_inplace(|v| v / denom);
        // Total variance for ratio computation
        let total_variance: f64 = (0..n_features).map(|i| cov[[i, i]]).sum();

        // Extract top-k eigenvalues/vectors via power iteration + deflation
        let mut rng = SmallRng::seed_from_u64(self.random_state.unwrap_or(0));
        let mut components = Array2::zeros((self.n_components, n_features));
        let mut explained_variance = Array1::zeros(self.n_components);

        for k in 0..self.n_components {
            let (eigenvalue, eigenvector) = Self::power_iteration(&cov, &mut rng, 1000, 1e-10);
            explained_variance[k] = eigenvalue;
            components.row_mut(k).assign(&eigenvector);
            Self::deflate(&mut cov, eigenvalue, &eigenvector);
        }

        // Compute explained variance ratio
        let explained_variance_ratio = if total_variance > 0.0 {
            explained_variance.mapv(|v| v / total_variance)
        } else {
            Array1::zeros(self.n_components)
        };

        self.components_ = Some(components);
        self.explained_variance_ = Some(explained_variance);
        self.explained_variance_ratio_ = Some(explained_variance_ratio);
        self.mean_ = Some(mean);

        Ok(())
    }

    pub fn transform(&self, X: &ArrayView2<f64>) -> Result<Array2<f64>, String> {
        let components = self.components_.as_ref().ok_or("PCA is not fitted yet")?;
        let mean = self.mean_.as_ref().unwrap();
        check_finite(X)?;

        if X.ncols() != mean.len() {
            return Err(format!(
                "Feature mismatch: expected {}, got {}",
                mean.len(),
                X.ncols()
            ));
        }

        let X_centered = Self::center(X, mean);

        // X_transformed = X_centered * components^T  (n_samples x n_components)
        let components_t = components.t();
        Ok(X_centered.dot(&components_t))
    }

    pub fn fit_transform(&mut self, X: &ArrayView2<f64>) -> Result<Array2<f64>, String> {
        self.fit(X)?;
        self.transform(X)
    }

    pub fn inverse_transform(&self, X_transformed: &ArrayView2<f64>) -> Result<Array2<f64>, String> {
        let components = self
            .components_
            .as_ref()
            .ok_or("PCA is not fitted yet")?;
        let mean = self.mean_.as_ref().unwrap();
        check_finite(X_transformed)?;

        if X_transformed.ncols() != self.n_components {
            return Err(format!(
                "Expected {} components, got {}",
                self.n_components,
                X_transformed.ncols()
            ));
        }

        // X_original  X_transformed * components + mean
        let mut X_original = X_transformed.dot(components);
        X_original
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .for_each(|mut row| {
                for j in 0..row.len() {
                    row[j] += mean[j];
                }
            });

        Ok(X_original)
    }
}

// ==========================================
// Tests
// ==========================================
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_pca_basic() {
        // Simple 2D data, reduce to 1 component
        let X = array![
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ];
        let mut pca = PCA::new(1, Some(42));
        pca.fit(&X.view()).unwrap();

        assert!(pca.components_.is_some());
        assert!(pca.explained_variance_.is_some());
        assert!(pca.explained_variance_ratio_.is_some());
        assert!(pca.mean_.is_some());

        let components = pca.components_.as_ref().unwrap();
        assert_eq!(components.shape(), &[1, 2]);

        // The explained variance ratio should be high for this correlated data
        let ratio = pca.explained_variance_ratio_.as_ref().unwrap();
        assert!(ratio[0] > 0.99);
    }

    #[test]
    fn test_pca_transform_inverse() {
        let X = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ];
        let mut pca = PCA::new(2, Some(0));
        pca.fit(&X.view()).unwrap();

        let X_t = pca.transform(&X.view()).unwrap();
        assert_eq!(X_t.shape(), &[4, 2]);

        let X_inv = pca.inverse_transform(&X_t.view()).unwrap();
        assert_eq!(X_inv.shape(), &[4, 3]);

        // Reconstruction should be close (2 components from 3 features)
        for i in 0..4 {
            for j in 0..3 {
                assert!(
                    (X_inv[[i, j]] - X[[i, j]]).abs() < 1e-6,
                    "Reconstruction error at [{}, {}]: {} vs {}",
                    i,
                    j,
                    X_inv[[i, j]],
                    X[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_pca_fit_transform() {
        let X = array![
            [2.5, 2.4],
            [0.5, 0.7],
            [2.2, 2.9],
            [1.9, 2.2],
            [3.1, 3.0],
            [2.3, 2.7],
            [2.0, 1.6],
            [1.0, 1.1],
            [1.5, 1.6],
            [1.1, 0.9],
        ];
        let mut pca = PCA::new(2, Some(42));
        let X_t = pca.fit_transform(&X.view()).unwrap();
        assert_eq!(X_t.shape(), &[10, 2]);

        let ratio = pca.explained_variance_ratio_.as_ref().unwrap();
        let total: f64 = ratio.iter().sum();
        assert!((total - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_pca_mean() {
        let X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let mut pca = PCA::new(1, Some(0));
        pca.fit(&X.view()).unwrap();

        let mean = pca.mean_.as_ref().unwrap();
        assert!((mean[0] - 3.0).abs() < 1e-10);
        assert!((mean[1] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_pca_errors() {
        let empty: Array2<f64> = Array2::zeros((0, 2));
        let mut pca = PCA::new(1, None);
        assert!(pca.fit(&empty.view()).is_err());

        let X = array![[1.0, 2.0], [3.0, 4.0]];
        let mut pca3 = PCA::new(3, None);
        assert!(pca3.fit(&X.view()).is_err()); // n_components > n_features

        // Not fitted
        let pca_unfitted = PCA::new(1, None);
        assert!(pca_unfitted.transform(&X.view()).is_err());
    }
}
