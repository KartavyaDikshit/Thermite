#![allow(non_snake_case)]
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rayon::prelude::*;
use std::collections::HashMap;

fn check_finite(X: &ArrayView2<f64>) -> Result<(), String> {
    if X.iter().any(|&val| !val.is_finite()) {
        return Err("Input contains NaN or infinity values".to_string());
    }
    Ok(())
}

fn euclidean_dist_sq(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).powi(2)).sum()
}

// ==========================================
// WeightType
// ==========================================
#[derive(Clone, Debug, PartialEq)]
pub enum WeightType {
    Uniform,
    Distance,
}

impl WeightType {
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "uniform" => Ok(WeightType::Uniform),
            "distance" => Ok(WeightType::Distance),
            _ => Err(format!("Unknown weight type: '{}'. Use 'uniform' or 'distance'.", s)),
        }
    }
}

// ==========================================
// KNeighborsClassifier
// ==========================================
pub struct KNeighborsClassifier {
    pub n_neighbors: usize,
    pub weights: WeightType,
    // Fitted data (stored for brute-force)
    X_train: Option<Array2<f64>>,
    y_train: Option<Array1<i64>>,
    classes_: Option<Vec<i64>>,
}

impl KNeighborsClassifier {
    pub fn new(n_neighbors: usize, weights: WeightType) -> Self {
        KNeighborsClassifier {
            n_neighbors,
            weights,
            X_train: None,
            y_train: None,
            classes_: None,
        }
    }

    pub fn fit(
        &mut self,
        X: &ArrayView2<f64>,
        y: &ArrayView1<i64>,
    ) -> Result<(), String> {
        if X.is_empty() {
            return Err("Input array is empty".to_string());
        }
        check_finite(X)?;

        if X.nrows() != y.len() {
            return Err(format!(
                "X has {} samples but y has {} labels",
                X.nrows(),
                y.len()
            ));
        }

        if X.nrows() < self.n_neighbors {
            return Err(format!(
                "n_neighbors={} is greater than n_samples={}",
                self.n_neighbors,
                X.nrows()
            ));
        }

        // Discover unique classes
        let mut classes: Vec<i64> = y.to_vec();
        classes.sort_unstable();
        classes.dedup();

        self.X_train = Some(X.to_owned());
        self.y_train = Some(y.to_owned());
        self.classes_ = Some(classes);

        Ok(())
    }

    /// For a single query point, find the k nearest neighbors.
    /// Returns Vec<(distance_sq, index)> sorted ascending by distance.
    fn find_neighbors(
        query: &[f64],
        X_train: &Array2<f64>,
        k: usize,
    ) -> Vec<(f64, usize)> {
        let n_train = X_train.nrows();

        // Compute all distances
        let mut dists: Vec<(f64, usize)> = (0..n_train)
            .map(|i| {
                let d = euclidean_dist_sq(query, X_train.row(i).as_slice().unwrap());
                (d, i)
            })
            .collect();

        // Partial sort to get top-k
        dists.select_nth_unstable_by(k - 1, |a, b| a.0.partial_cmp(&b.0).unwrap());
        dists.truncate(k);
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        dists
    }

    pub fn predict(&self, X: &ArrayView2<f64>) -> Result<Array1<i64>, String> {
        let X_train = self.X_train.as_ref().ok_or("KNeighborsClassifier is not fitted yet")?;
        let y_train = self.y_train.as_ref().unwrap();
        check_finite(X)?;

        if X.ncols() != X_train.ncols() {
            return Err(format!(
                "Feature mismatch: expected {}, got {}",
                X_train.ncols(),
                X.ncols()
            ));
        }

        let k = self.n_neighbors;
        let weights = &self.weights;

        let predictions: Vec<i64> = X
            .axis_iter(Axis(0))
            .into_par_iter()
            .map(|row| {
                let query = row.as_slice().unwrap();
                let neighbors = Self::find_neighbors(query, X_train, k);

                // Weighted vote
                let mut votes: HashMap<i64, f64> = HashMap::new();
                for &(dist_sq, idx) in &neighbors {
                    let label = y_train[idx];
                    let weight = match weights {
                        WeightType::Uniform => 1.0,
                        WeightType::Distance => {
                            let dist = dist_sq.sqrt();
                            if dist < 1e-15 {
                                1e15 // Essentially infinite weight for exact match
                            } else {
                                1.0 / dist
                            }
                        }
                    };
                    *votes.entry(label).or_insert(0.0) += weight;
                }

                // Find class with max weight
                votes
                    .into_iter()
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .unwrap()
                    .0
            })
            .collect();

        Ok(Array1::from(predictions))
    }

    pub fn predict_proba(&self, X: &ArrayView2<f64>) -> Result<Array2<f64>, String> {
        let X_train = self.X_train.as_ref().ok_or("KNeighborsClassifier is not fitted yet")?;
        let y_train = self.y_train.as_ref().unwrap();
        let classes = self.classes_.as_ref().unwrap();
        check_finite(X)?;

        if X.ncols() != X_train.ncols() {
            return Err(format!(
                "Feature mismatch: expected {}, got {}",
                X_train.ncols(),
                X.ncols()
            ));
        }

        let k = self.n_neighbors;
        let n_classes = classes.len();
        let weights = &self.weights;

        // Build class-to-index map
        let class_idx: HashMap<i64, usize> = classes
            .iter()
            .enumerate()
            .map(|(i, &c)| (c, i))
            .collect();

        let probas: Vec<Vec<f64>> = X
            .axis_iter(Axis(0))
            .into_par_iter()
            .map(|row| {
                let query = row.as_slice().unwrap();
                let neighbors = Self::find_neighbors(query, X_train, k);

                let mut class_weights = vec![0.0f64; n_classes];
                let mut total_weight = 0.0;

                for &(dist_sq, idx) in &neighbors {
                    let label = y_train[idx];
                    let weight = match weights {
                        WeightType::Uniform => 1.0,
                        WeightType::Distance => {
                            let dist = dist_sq.sqrt();
                            if dist < 1e-15 {
                                1e15
                            } else {
                                1.0 / dist
                            }
                        }
                    };
                    if let Some(&ci) = class_idx.get(&label) {
                        class_weights[ci] += weight;
                    }
                    total_weight += weight;
                }

                // Normalize to probabilities
                if total_weight > 0.0 {
                    for w in class_weights.iter_mut() {
                        *w /= total_weight;
                    }
                }

                class_weights
            })
            .collect();

        let n_samples = X.nrows();
        let mut result = Array2::zeros((n_samples, n_classes));
        for (i, proba_row) in probas.into_iter().enumerate() {
            for (j, val) in proba_row.into_iter().enumerate() {
                result[[i, j]] = val;
            }
        }

        Ok(result)
    }

    pub fn score(
        &self,
        X: &ArrayView2<f64>,
        y: &ArrayView1<i64>,
    ) -> Result<f64, String> {
        let predictions = self.predict(X)?;

        if predictions.len() != y.len() {
            return Err("Prediction and y length mismatch".to_string());
        }

        let correct = predictions
            .iter()
            .zip(y.iter())
            .filter(|(&pred, &true_label)| pred == true_label)
            .count();

        Ok(correct as f64 / y.len() as f64)
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
    fn test_knn_basic_uniform() {
        // Simple 2-class problem
        let X_train = array![
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [10.0, 10.0],
            [11.0, 10.0],
            [10.0, 11.0],
        ];
        let y_train = array![0i64, 0, 0, 1, 1, 1];

        let mut knn = KNeighborsClassifier::new(3, WeightType::Uniform);
        knn.fit(&X_train.view(), &y_train.view()).unwrap();

        let X_test = array![[0.5, 0.5], [10.5, 10.5]];
        let pred = knn.predict(&X_test.view()).unwrap();
        assert_eq!(pred[0], 0);
        assert_eq!(pred[1], 1);
    }

    #[test]
    fn test_knn_distance_weighted() {
        let X_train = array![
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ];
        let y_train = array![0i64, 0, 1];

        let mut knn = KNeighborsClassifier::new(3, WeightType::Distance);
        knn.fit(&X_train.view(), &y_train.view()).unwrap();

        // Point very close to [0,0]  class 0
        let X_test = array![[0.01, 0.0]];
        let pred = knn.predict(&X_test.view()).unwrap();
        assert_eq!(pred[0], 0);
    }

    #[test]
    fn test_knn_predict_proba() {
        let X_train = array![
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ];
        let y_train = array![0i64, 0, 1];

        let mut knn = KNeighborsClassifier::new(3, WeightType::Uniform);
        knn.fit(&X_train.view(), &y_train.view()).unwrap();

        let X_test = array![[0.5, 0.0]];
        let proba = knn.predict_proba(&X_test.view()).unwrap();
        assert_eq!(proba.shape(), &[1, 2]);

        // 2 out of 3 neighbors are class 0
        assert!((proba[[0, 0]] - 2.0 / 3.0).abs() < 1e-10);
        assert!((proba[[0, 1]] - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_knn_score() {
        let X_train = array![
            [0.0, 0.0],
            [1.0, 0.0],
            [10.0, 10.0],
            [11.0, 10.0],
        ];
        let y_train = array![0i64, 0, 1, 1];

        let mut knn = KNeighborsClassifier::new(2, WeightType::Uniform);
        knn.fit(&X_train.view(), &y_train.view()).unwrap();

        let score = knn.score(&X_train.view(), &y_train.view()).unwrap();
        assert!((score - 1.0).abs() < 1e-10); // Perfect on training data
    }

    #[test]
    fn test_knn_errors() {
        let empty: Array2<f64> = Array2::zeros((0, 2));
        let y_empty: Array1<i64> = Array1::zeros(0);
        let mut knn = KNeighborsClassifier::new(3, WeightType::Uniform);
        assert!(knn.fit(&empty.view(), &y_empty.view()).is_err());

        // Mismatched X and y
        let X = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0i64];
        assert!(knn.fit(&X.view(), &y.view()).is_err());

        // n_neighbors > n_samples
        let X2 = array![[1.0, 2.0]];
        let y2 = array![0i64];
        let mut knn5 = KNeighborsClassifier::new(5, WeightType::Uniform);
        assert!(knn5.fit(&X2.view(), &y2.view()).is_err());

        // Not fitted
        let unfitted = KNeighborsClassifier::new(3, WeightType::Uniform);
        assert!(unfitted.predict(&X.view()).is_err());
    }

    #[test]
    fn test_knn_multiclass() {
        let X_train = array![
            [0.0, 0.0],
            [0.5, 0.0],
            [5.0, 5.0],
            [5.5, 5.0],
            [10.0, 0.0],
            [10.5, 0.0],
        ];
        let y_train = array![0i64, 0, 1, 1, 2, 2];

        let mut knn = KNeighborsClassifier::new(2, WeightType::Uniform);
        knn.fit(&X_train.view(), &y_train.view()).unwrap();

        let X_test = array![[0.2, 0.0], [5.2, 5.0], [10.2, 0.0]];
        let pred = knn.predict(&X_test.view()).unwrap();
        assert_eq!(pred[0], 0);
        assert_eq!(pred[1], 1);
        assert_eq!(pred[2], 2);

        let proba = knn.predict_proba(&X_test.view()).unwrap();
        assert_eq!(proba.shape(), &[3, 3]); // 3 samples, 3 classes
    }

    #[test]
    fn test_weight_type_from_str() {
        assert_eq!(WeightType::from_str("uniform").unwrap(), WeightType::Uniform);
        assert_eq!(WeightType::from_str("distance").unwrap(), WeightType::Distance);
        assert!(WeightType::from_str("invalid").is_err());
    }
}
