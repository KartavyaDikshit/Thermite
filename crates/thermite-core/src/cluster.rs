#![allow(non_snake_case)]
use ndarray::{Array2, ArrayView2, Axis};
use rand::prelude::*;
use rand::rngs::SmallRng;
use rayon::prelude::*;
use sprs::CsMat;

// ==========================================
// Helper: Euclidean distance squared
// ==========================================
fn euclidean_dist_sq(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).powi(2)).sum()
}

fn check_finite(X: &ArrayView2<f64>) -> Result<(), String> {
    if X.iter().any(|&val| !val.is_finite()) {
        return Err("Input contains NaN or infinity values".to_string());
    }
    Ok(())
}

// ==========================================
// KMeans
// ==========================================
pub struct KMeans {
    pub n_clusters: usize,
    pub max_iter: usize,
    pub tol: f64,
    pub n_init: usize,
    pub random_state: Option<u64>,
    // Fitted attributes
    pub cluster_centers_: Option<Array2<f64>>,
    pub labels_: Option<Vec<usize>>,
    pub inertia_: Option<f64>,
    pub n_iter_: Option<usize>,
}

impl KMeans {
    pub fn new(
        n_clusters: usize,
        max_iter: usize,
        tol: f64,
        n_init: usize,
        random_state: Option<u64>,
    ) -> Self {
        KMeans {
            n_clusters,
            max_iter,
            tol,
            n_init,
            random_state,
            cluster_centers_: None,
            labels_: None,
            inertia_: None,
            n_iter_: None,
        }
    }

    /// K-means++ initialization: pick first center uniformly at random,
    /// then choose subsequent centers with probability proportional to D(x)^2.
    fn kmeans_pp_init(&self, X: &ArrayView2<f64>, rng: &mut SmallRng) -> Array2<f64> {
        let n_samples = X.nrows();
        let n_features = X.ncols();
        let mut centers = Array2::zeros((self.n_clusters, n_features));

        // Pick first center uniformly at random
        let first_idx = rng.gen_range(0..n_samples);
        centers.row_mut(0).assign(&X.row(first_idx));

        // Distance from each point to its nearest center so far
        let mut min_dists = vec![f64::INFINITY; n_samples];

        for c in 1..self.n_clusters {
            // Update min_dists with the last added center
            let prev_center = centers.row(c - 1);
            for i in 0..n_samples {
                let d = euclidean_dist_sq(X.row(i).as_slice().unwrap(), prev_center.as_slice().unwrap());
                if d < min_dists[i] {
                    min_dists[i] = d;
                }
            }

            // Cumulative distribution
            let total: f64 = min_dists.iter().sum();
            if total == 0.0 {
                // All remaining points coincide with existing centers
                let idx = rng.gen_range(0..n_samples);
                centers.row_mut(c).assign(&X.row(idx));
                continue;
            }

            let threshold = rng.gen::<f64>() * total;
            let mut cumsum = 0.0;
            let mut chosen = n_samples - 1;
            for (i, &d) in min_dists.iter().enumerate() {
                cumsum += d;
                if cumsum >= threshold {
                    chosen = i;
                    break;
                }
            }
            centers.row_mut(c).assign(&X.row(chosen));
        }

        centers
    }

    /// Assign each sample to the nearest cluster center.
    /// Uses the identity ||x - c||^2 = ||x||^2 - 2*x.c + ||c||^2
    /// to compute all distances via a single matrix multiplication.
    fn assign_labels(X: &ArrayView2<f64>, centers: &Array2<f64>) -> (Vec<usize>, f64) {
        let n_samples = X.nrows();
        let n_clusters = centers.nrows();

        let results: Vec<(usize, f64)> = X.axis_iter(Axis(0))
            .into_par_iter()
            .map(|row| {
                let mut best_label = 0;
                let mut best_dist = f64::INFINITY;
                let row_slice = row.as_slice().unwrap();

                for k in 0..n_clusters {
                    let c_row = centers.row(k);
                    let c_slice = c_row.as_slice().unwrap();
                    let mut d = 0.0;
                    for j in 0..row_slice.len() {
                        let diff = row_slice[j] - c_slice[j];
                        d += diff * diff;
                    }
                    if d < best_dist {
                        best_dist = d;
                        best_label = k;
                    }
                }
                (best_label, best_dist)
            })
            .collect();

        let mut labels = Vec::with_capacity(n_samples);
        let mut inertia = 0.0;
        for (l, d) in results {
            labels.push(l);
            inertia += d;
        }
        (labels, inertia)
    }

    /// Recompute cluster centers from assignments using vectorized row ops.
    fn update_centers(
        X: &ArrayView2<f64>,
        labels: &[usize],
        n_clusters: usize,
        n_features: usize,
    ) -> Array2<f64> {
        let mut centers = Array2::zeros((n_clusters, n_features));
        let mut counts = vec![0usize; n_clusters];

        for (i, &label) in labels.iter().enumerate() {
            counts[label] += 1;
            let x_row = X.row(i);
            let mut c_row = centers.row_mut(label);
            c_row += &x_row;
        }

        for k in 0..n_clusters {
            if counts[k] > 0 {
                let c = counts[k] as f64;
                centers.row_mut(k).mapv_inplace(|v| v / c);
            }
        }

        centers
    }

    /// Run one full Lloyd's iteration sequence.
    fn run_single(
        &self,
        X: &ArrayView2<f64>,
        rng: &mut SmallRng,
    ) -> (Array2<f64>, Vec<usize>, f64, usize) {
        let n_features = X.ncols();
        let mut centers = self.kmeans_pp_init(X, rng);
        let mut labels;
        let mut inertia;
        let mut n_iter = 0;

        loop {
            let (new_labels, new_inertia) = Self::assign_labels(X, &centers);
            labels = new_labels;
            inertia = new_inertia;
            n_iter += 1;

            let new_centers = Self::update_centers(X, &labels, self.n_clusters, n_features);

            // Check convergence: maximum center shift
            let shift: f64 = centers
                .axis_iter(Axis(0))
                .into_par_iter()
                .zip(new_centers.axis_iter(Axis(0)).into_par_iter())
                .map(|(old, new)| {
                    euclidean_dist_sq(old.as_slice().unwrap(), new.as_slice().unwrap())
                })
                .reduce(|| 0.0f64, f64::max);

            centers = new_centers;

            if shift <= self.tol || n_iter >= self.max_iter {
                break;
            }
        }

        (centers, labels, inertia, n_iter)
    }

    pub fn fit(&mut self, X: &ArrayView2<f64>) -> Result<(), String> {
        if X.is_empty() {
            return Err("Input array is empty".to_string());
        }
        check_finite(X)?;

        let n_samples = X.nrows();
        if n_samples < self.n_clusters {
            return Err(format!(
                "n_samples={} should be >= n_clusters={}",
                n_samples, self.n_clusters
            ));
        }

        let base_seed = self.random_state.unwrap_or(0);
        let mut best_centers = None;
        let mut best_labels = None;
        let mut best_inertia = f64::INFINITY;
        let mut best_n_iter = 0;

        for init in 0..self.n_init {
            let mut rng = SmallRng::seed_from_u64(base_seed.wrapping_add(init as u64));
            let (centers, labels, inertia, n_iter) = self.run_single(X, &mut rng);

            if inertia < best_inertia {
                best_inertia = inertia;
                best_centers = Some(centers);
                best_labels = Some(labels);
                best_n_iter = n_iter;
            }
        }

        self.cluster_centers_ = best_centers;
        self.labels_ = best_labels;
        self.inertia_ = Some(best_inertia);
        self.n_iter_ = Some(best_n_iter);

        Ok(())
    }

    pub fn predict(&self, X: &ArrayView2<f64>) -> Result<Vec<usize>, String> {
        let centers = self
            .cluster_centers_
            .as_ref()
            .ok_or("KMeans is not fitted yet")?;
        check_finite(X)?;

        if X.ncols() != centers.ncols() {
            return Err(format!(
                "Feature mismatch: expected {}, got {}",
                centers.ncols(),
                X.ncols()
            ));
        }

        let (labels, _) = Self::assign_labels(X, centers);
        Ok(labels)
    }

    pub fn fit_predict(&mut self, X: &ArrayView2<f64>) -> Result<Vec<usize>, String> {
        self.fit(X)?;
        Ok(self.labels_.clone().unwrap())
    }

    fn kmeans_pp_init_sparse(&self, X: &CsMat<f64>, rng: &mut SmallRng) -> Array2<f64> {
        let n_samples = X.rows();
        let n_features = X.cols();
        let mut centers = Array2::zeros((self.n_clusters, n_features));

        let first_idx = rng.gen_range(0..n_samples);
        let first_row = X.outer_view(first_idx).unwrap();
        for (col_idx, &val) in first_row.iter() {
            centers[[0, col_idx]] = val;
        }

        let mut min_dists = vec![f64::INFINITY; n_samples];

        for c in 1..self.n_clusters {
            let prev_center = centers.row(c - 1);
            let prev_center_slice = prev_center.as_slice().unwrap();
            let mut norm_c_sq = 0.0;
            for &val in prev_center_slice {
                norm_c_sq += val * val;
            }

            for i in 0..n_samples {
                let row = X.outer_view(i).unwrap();
                let mut x_dot_c = 0.0;
                let mut norm_x_sq = 0.0;
                for (col_idx, &val) in row.iter() {
                    x_dot_c += val * prev_center_slice[col_idx];
                    norm_x_sq += val * val;
                }
                let mut d = norm_x_sq - 2.0 * x_dot_c + norm_c_sq;
                if d < 0.0 { d = 0.0; }
                if d < min_dists[i] {
                    min_dists[i] = d;
                }
            }

            let total: f64 = min_dists.iter().sum();
            if total == 0.0 {
                let idx = rng.gen_range(0..n_samples);
                let row = X.outer_view(idx).unwrap();
                for (col_idx, &val) in row.iter() {
                    centers[[c, col_idx]] = val;
                }
                continue;
            }

            let threshold = rng.gen::<f64>() * total;
            let mut cumsum = 0.0;
            let mut chosen = n_samples - 1;
            for (i, &d) in min_dists.iter().enumerate() {
                cumsum += d;
                if cumsum >= threshold {
                    chosen = i;
                    break;
                }
            }
            let chosen_row = X.outer_view(chosen).unwrap();
            for (col_idx, &val) in chosen_row.iter() {
                centers[[c, col_idx]] = val;
            }
        }

        centers
    }

    fn assign_labels_sparse(X: &CsMat<f64>, centers: &Array2<f64>) -> (Vec<usize>, f64) {
        let n_samples = X.rows();
        let n_clusters = centers.nrows();

        let mut center_norms_sq = vec![0.0; n_clusters];
        for k in 0..n_clusters {
            let mut norm_sq = 0.0;
            for &val in centers.row(k) {
                norm_sq += val * val;
            }
            center_norms_sq[k] = norm_sq;
        }

        let results: Vec<(usize, f64)> = (0..n_samples)
            .into_par_iter()
            .map(|i| {
                let row = X.outer_view(i).unwrap();
                let mut best_label = 0;
                let mut best_dist = f64::INFINITY;
                
                let mut norm_x_sq = 0.0;
                for (_, &val) in row.iter() {
                    norm_x_sq += val * val;
                }

                for k in 0..n_clusters {
                    let row_view = centers.row(k);
                    let c_slice = row_view.as_slice().unwrap();
                    let mut x_dot_c = 0.0;
                    for (col_idx, &val) in row.iter() {
                        x_dot_c += val * c_slice[col_idx];
                    }
                    let mut d = norm_x_sq - 2.0 * x_dot_c + center_norms_sq[k];
                    if d < 0.0 { d = 0.0; }
                    
                    if d < best_dist {
                        best_dist = d;
                        best_label = k;
                    }
                }
                (best_label, best_dist)
            })
            .collect();

        let mut labels = Vec::with_capacity(n_samples);
        let mut inertia = 0.0;
        for (l, d) in results {
            labels.push(l);
            inertia += d;
        }
        (labels, inertia)
    }

    fn update_centers_sparse(
        X: &CsMat<f64>,
        labels: &[usize],
        n_clusters: usize,
        n_features: usize,
    ) -> Array2<f64> {
        let mut centers = Array2::zeros((n_clusters, n_features));
        let mut counts = vec![0usize; n_clusters];

        for i in 0..X.rows() {
            let label = labels[i];
            counts[label] += 1;
            let row = X.outer_view(i).unwrap();
            for (col_idx, &val) in row.iter() {
                centers[[label, col_idx]] += val;
            }
        }

        for k in 0..n_clusters {
            if counts[k] > 0 {
                let c = counts[k] as f64;
                centers.row_mut(k).mapv_inplace(|v| v / c);
            }
        }

        centers
    }

    fn run_single_sparse(
        &self,
        X: &CsMat<f64>,
        rng: &mut SmallRng,
    ) -> (Array2<f64>, Vec<usize>, f64, usize) {
        let n_features = X.cols();
        let mut centers = self.kmeans_pp_init_sparse(X, rng);
        let mut labels;
        let mut inertia;
        let mut n_iter = 0;

        loop {
            let (new_labels, new_inertia) = Self::assign_labels_sparse(X, &centers);
            labels = new_labels;
            inertia = new_inertia;
            n_iter += 1;

            let new_centers = Self::update_centers_sparse(X, &labels, self.n_clusters, n_features);

            let shift: f64 = centers
                .axis_iter(Axis(0))
                .into_par_iter()
                .zip(new_centers.axis_iter(Axis(0)).into_par_iter())
                .map(|(old, new)| {
                    euclidean_dist_sq(old.as_slice().unwrap(), new.as_slice().unwrap())
                })
                .reduce(|| 0.0f64, f64::max);

            centers = new_centers;

            if shift <= self.tol || n_iter >= self.max_iter {
                break;
            }
        }

        (centers, labels, inertia, n_iter)
    }

    pub fn fit_sparse(&mut self, X: &CsMat<f64>) -> Result<(), String> {
        if X.rows() == 0 || X.cols() == 0 {
            return Err("Input array is empty".to_string());
        }

        let n_samples = X.rows();
        if n_samples < self.n_clusters {
            return Err(format!(
                "n_samples={} should be >= n_clusters={}",
                n_samples, self.n_clusters
            ));
        }

        let base_seed = self.random_state.unwrap_or(0);
        let mut best_centers = None;
        let mut best_labels = None;
        let mut best_inertia = f64::INFINITY;
        let mut best_n_iter = 0;

        for init in 0..self.n_init {
            let mut rng = SmallRng::seed_from_u64(base_seed.wrapping_add(init as u64));
            let (centers, labels, inertia, n_iter) = self.run_single_sparse(X, &mut rng);

            if inertia < best_inertia {
                best_inertia = inertia;
                best_centers = Some(centers);
                best_labels = Some(labels);
                best_n_iter = n_iter;
            }
        }

        self.cluster_centers_ = best_centers;
        self.labels_ = best_labels;
        self.inertia_ = Some(best_inertia);
        self.n_iter_ = Some(best_n_iter);

        Ok(())
    }

    pub fn predict_sparse(&self, X: &CsMat<f64>) -> Result<Vec<usize>, String> {
        let centers = self
            .cluster_centers_
            .as_ref()
            .ok_or("KMeans is not fitted yet")?;

        if X.cols() != centers.ncols() {
            return Err(format!(
                "Feature mismatch: expected {}, got {}",
                centers.ncols(),
                X.cols()
            ));
        }

        let (labels, _) = Self::assign_labels_sparse(X, centers);
        Ok(labels)
    }
}

// ==========================================
// DBSCAN
// ==========================================
pub struct DBSCAN {
    pub eps: f64,
    pub min_samples: usize,
    // Fitted attributes
    pub labels_: Option<Vec<i64>>,
    pub core_sample_indices_: Option<Vec<usize>>,
}

impl DBSCAN {
    pub fn new(eps: f64, min_samples: usize) -> Self {
        DBSCAN {
            eps,
            min_samples,
            labels_: None,
            core_sample_indices_: None,
        }
    }

    pub fn fit(&mut self, X: &ArrayView2<f64>) -> Result<(), String> {
        if X.is_empty() {
            return Err("Input array is empty".to_string());
        }
        check_finite(X)?;

        let n_samples = X.nrows();
        let eps_sq = self.eps * self.eps;

        // Step 1: Find neighbors for each point (parallel)
        let neighborhoods: Vec<Vec<usize>> = (0..n_samples)
            .into_par_iter()
            .map(|i| {
                let row_i = X.row(i);
                let slice_i = row_i.as_slice().unwrap();
                let mut neighbors = Vec::new();
                for j in 0..n_samples {
                    let d = euclidean_dist_sq(slice_i, X.row(j).as_slice().unwrap());
                    if d <= eps_sq {
                        neighbors.push(j);
                    }
                }
                neighbors
            })
            .collect();

        // Step 2: Identify core samples
        let is_core: Vec<bool> = neighborhoods
            .iter()
            .map(|n| n.len() >= self.min_samples)
            .collect();

        let core_sample_indices: Vec<usize> = is_core
            .iter()
            .enumerate()
            .filter_map(|(i, &c)| if c { Some(i) } else { None })
            .collect();

        // Step 3: BFS/DFS to form clusters
        let mut labels = vec![-1i64; n_samples];
        let mut cluster_id: i64 = 0;

        for i in 0..n_samples {
            if labels[i] != -1 || !is_core[i] {
                continue;
            }

            // Start a new cluster
            labels[i] = cluster_id;
            let mut stack = vec![i];

            while let Some(current) = stack.pop() {
                for &neighbor in &neighborhoods[current] {
                    if labels[neighbor] == -1 {
                        labels[neighbor] = cluster_id;
                        if is_core[neighbor] {
                            stack.push(neighbor);
                        }
                    }
                }
            }

            cluster_id += 1;
        }

        self.labels_ = Some(labels);
        self.core_sample_indices_ = Some(core_sample_indices);

        Ok(())
    }

    pub fn fit_predict(&mut self, X: &ArrayView2<f64>) -> Result<Vec<i64>, String> {
        self.fit(X)?;
        Ok(self.labels_.clone().unwrap())
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
    fn test_kmeans_basic() {
        // Two obvious clusters
        let X = array![
            [1.0, 1.0],
            [1.5, 1.5],
            [1.0, 1.5],
            [10.0, 10.0],
            [10.5, 10.5],
            [10.0, 10.5],
        ];
        let mut km = KMeans::new(2, 100, 1e-4, 3, Some(42));
        km.fit(&X.view()).unwrap();

        let labels = km.labels_.as_ref().unwrap();
        // Points in the same cluster should share a label
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[0], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[3], labels[5]);
        // The two clusters should differ
        assert_ne!(labels[0], labels[3]);

        assert!(km.inertia_.unwrap() < 5.0);
        assert!(km.n_iter_.unwrap() > 0);
    }

    #[test]
    fn test_kmeans_predict() {
        let X = array![
            [0.0, 0.0],
            [0.5, 0.5],
            [10.0, 10.0],
            [10.5, 10.5],
        ];
        let mut km = KMeans::new(2, 100, 1e-4, 1, Some(0));
        km.fit(&X.view()).unwrap();

        let new_point = array![[0.1, 0.1], [9.9, 9.9]];
        let pred = km.predict(&new_point.view()).unwrap();
        // The new point near [0,0] should be same cluster as X[0]
        let labels = km.labels_.as_ref().unwrap();
        assert_eq!(pred[0], labels[0]);
        assert_eq!(pred[1], labels[2]);
    }

    #[test]
    fn test_kmeans_fit_predict() {
        let X = array![[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0]];
        let mut km = KMeans::new(2, 100, 1e-4, 1, Some(42));
        let labels = km.fit_predict(&X.view()).unwrap();
        assert_eq!(labels.len(), 4);
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
    }

    #[test]
    fn test_kmeans_errors() {
        let empty: Array2<f64> = Array2::zeros((0, 2));
        let mut km = KMeans::new(2, 100, 1e-4, 1, None);
        assert!(km.fit(&empty.view()).is_err());

        let X = array![[1.0, 2.0]];
        assert!(km.fit(&X.view()).is_err()); // 1 sample < 2 clusters
    }

    #[test]
    fn test_dbscan_basic() {
        let X = array![
            [1.0, 1.0],
            [1.1, 1.0],
            [1.0, 1.1],
            [10.0, 10.0],
            [10.1, 10.0],
            [10.0, 10.1],
            [100.0, 100.0], // noise point
        ];
        let mut db = DBSCAN::new(0.5, 2);
        let labels = db.fit_predict(&X.view()).unwrap();

        // First cluster
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[0], labels[2]);
        // Second cluster
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[3], labels[5]);
        // Two clusters differ
        assert_ne!(labels[0], labels[3]);
        // Noise point
        assert_eq!(labels[6], -1);

        // Core samples should not include the noise point
        let cores = db.core_sample_indices_.as_ref().unwrap();
        assert!(!cores.contains(&6));
    }

    #[test]
    fn test_dbscan_all_noise() {
        let X = array![[0.0, 0.0], [100.0, 100.0], [200.0, 200.0]];
        let mut db = DBSCAN::new(0.5, 2);
        let labels = db.fit_predict(&X.view()).unwrap();
        assert!(labels.iter().all(|&l| l == -1));
    }

    #[test]
    fn test_dbscan_single_cluster() {
        let X = array![[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1]];
        let mut db = DBSCAN::new(0.5, 2);
        let labels = db.fit_predict(&X.view()).unwrap();
        // All should be in same cluster
        let first = labels[0];
        assert!(first >= 0);
        assert!(labels.iter().all(|&l| l == first));
    }
}
