#![allow(non_snake_case)]

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rand::prelude::*;
use rand::rngs::SmallRng;
use rayon::prelude::*;
use crate::tree::{DecisionTreeClassifier, DecisionTreeRegressor};

// ==========================================
// RandomForestClassifier
// ==========================================
pub struct RandomForestClassifier {
    pub n_estimators: usize,
    pub max_depth: Option<usize>,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    pub max_features: Option<usize>, // defaults to sqrt
    pub random_state: Option<u64>,
    pub estimators_: Vec<DecisionTreeClassifier>,
    pub classes_: Option<Vec<f64>>,
    pub categorical_features: Vec<usize>,
}

impl RandomForestClassifier {
    pub fn new(
        n_estimators: usize,
        max_depth: Option<usize>,
        min_samples_split: usize,
        min_samples_leaf: usize,
        max_features: Option<usize>,
        random_state: Option<u64>,
    ) -> Self {
        RandomForestClassifier {
            n_estimators,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            max_features,
            random_state,
            estimators_: Vec::new(),
            classes_: None,
            categorical_features: Vec::new(),
        }
    }

    pub fn fit(&mut self, X: &ArrayView2<f64>, y: &ArrayView1<f64>) -> Result<(), String> {
        let n_samples = X.nrows();
        let n_features = X.ncols();
        let base_seed = self.random_state.unwrap_or(0);

        let mut classes = y.to_vec();
        classes.sort_by(|a, b| a.partial_cmp(b).unwrap());
        classes.dedup();
        self.classes_ = Some(classes);

        // Train trees in parallel using Rayon
        let estimators: Vec<DecisionTreeClassifier> = (0..self.n_estimators)
            .into_par_iter()
            .map(|i| {
                let seed = base_seed.wrapping_add(i as u64);
                let mut rng = SmallRng::seed_from_u64(seed);
                
                // Bootstrap sample using fast index mapping
                let mut indices = Vec::with_capacity(n_samples);
                for _ in 0..n_samples {
                    indices.push(rng.gen_range(0..n_samples));
                }
                let X_boot = X.select(Axis(0), &indices);
                let y_boot = y.select(Axis(0), &indices);

                // Default max_features to sqrt(n_features) for classification
                let max_feat = self.max_features.unwrap_or_else(|| (n_features as f64).sqrt().ceil() as usize).max(1);

                let mut tree = DecisionTreeClassifier::new(
                    self.max_depth,
                    self.min_samples_split,
                    self.min_samples_leaf,
                    Some(max_feat),
                    Some(seed),
                );
                tree.categorical_features = self.categorical_features.clone();
                
                tree.fit(&X_boot.view(), y_boot.as_slice().unwrap());
                tree
            })
            .collect();

        self.estimators_ = estimators;
        Ok(())
    }

    pub fn predict(&self, X: &ArrayView2<f64>) -> Result<Array1<f64>, String> {
        if self.estimators_.is_empty() {
            return Err("Model is not fitted".to_string());
        }

        let n_samples = X.nrows();
        let n_estimators = self.estimators_.len();
        
        // Predict with all trees in parallel
        let all_preds: Vec<Array1<f64>> = self.estimators_
            .par_iter()
            .map(|tree| Array1::from(tree.predict(X)))
            .collect();

        // Majority vote
        let mut final_preds = Array1::<f64>::zeros(n_samples);
        for i in 0..n_samples {
            let mut counts = std::collections::HashMap::new();
            for j in 0..n_estimators {
                let p = all_preds[j][i].to_bits(); // exact float matching for labels
                *counts.entry(p).or_insert(0) += 1;
            }
            let best = counts.into_iter().max_by_key(|&(_, count)| count).unwrap().0;
            final_preds[i] = f64::from_bits(best);
        }

        Ok(final_preds)
    }
}

// ==========================================
// RandomForestRegressor
// ==========================================
pub struct RandomForestRegressor {
    pub n_estimators: usize,
    pub max_depth: Option<usize>,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    pub max_features: Option<usize>, // defaults to n_features
    pub random_state: Option<u64>,
    pub estimators_: Vec<DecisionTreeRegressor>,
    pub categorical_features: Vec<usize>,
}

impl RandomForestRegressor {
    pub fn new(
        n_estimators: usize,
        max_depth: Option<usize>,
        min_samples_split: usize,
        min_samples_leaf: usize,
        max_features: Option<usize>,
        random_state: Option<u64>,
    ) -> Self {
        RandomForestRegressor {
            n_estimators,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            max_features,
            random_state,
            estimators_: Vec::new(),
            categorical_features: Vec::new(),
        }
    }

    pub fn fit(&mut self, X: &ArrayView2<f64>, y: &ArrayView1<f64>) -> Result<(), String> {
        let n_samples = X.nrows();
        let n_features = X.ncols();
        let base_seed = self.random_state.unwrap_or(0);

        // Train trees in parallel using Rayon
        let estimators: Vec<DecisionTreeRegressor> = (0..self.n_estimators)
            .into_par_iter()
            .map(|i| {
                let seed = base_seed.wrapping_add(i as u64);
                let mut rng = SmallRng::seed_from_u64(seed);
                
                // Bootstrap sample using fast index mapping
                let mut indices = Vec::with_capacity(n_samples);
                for _ in 0..n_samples {
                    indices.push(rng.gen_range(0..n_samples));
                }
                let X_boot = X.select(Axis(0), &indices);
                let y_boot = y.select(Axis(0), &indices);

                // Default max_features to n_features for regression
                let max_feat = self.max_features.unwrap_or(n_features).max(1);

                let mut tree = DecisionTreeRegressor::new(
                    self.max_depth,
                    self.min_samples_split,
                    self.min_samples_leaf,
                    Some(max_feat),
                    Some(seed),
                );
                tree.categorical_features = self.categorical_features.clone();
                
                tree.fit(&X_boot.view(), y_boot.as_slice().unwrap());
                tree
            })
            .collect();

        self.estimators_ = estimators;
        Ok(())
    }

    pub fn predict(&self, X: &ArrayView2<f64>) -> Result<Array1<f64>, String> {
        if self.estimators_.is_empty() {
            return Err("Model is not fitted".to_string());
        }

        let n_samples = X.nrows();
        let n_estimators = self.estimators_.len();
        
        // Predict with all trees in parallel
        let all_preds: Vec<Array1<f64>> = self.estimators_
            .par_iter()
            .map(|tree| Array1::from(tree.predict(X)))
            .collect();

        // Average predictions
        let mut final_preds = Array1::<f64>::zeros(n_samples);
        for i in 0..n_samples {
            let mut sum = 0.0;
            for j in 0..n_estimators {
                sum += all_preds[j][i];
            }
            final_preds[i] = sum / (n_estimators as f64);
        }

        Ok(final_preds)
    }
}

// ==========================================
// GradientBoostingRegressor
// ==========================================
pub struct GradientBoostingRegressor {
    pub n_estimators: usize,
    pub learning_rate: f64,
    pub max_depth: Option<usize>,
    pub random_state: Option<u64>,
    pub initial_prediction_: f64,
    pub estimators_: Vec<DecisionTreeRegressor>,
    pub categorical_features: Vec<usize>,
}

impl GradientBoostingRegressor {
    pub fn new(
        n_estimators: usize,
        learning_rate: f64,
        max_depth: Option<usize>,
        random_state: Option<u64>,
    ) -> Self {
        GradientBoostingRegressor {
            n_estimators,
            learning_rate,
            max_depth,
            random_state,
            initial_prediction_: 0.0,
            estimators_: Vec::new(),
            categorical_features: Vec::new(),
        }
    }

    pub fn fit(&mut self, X: &ArrayView2<f64>, y: &ArrayView1<f64>) -> Result<(), String> {
        let n_samples = X.nrows();
        let base_seed = self.random_state.unwrap_or(0);

        // F0 is the mean of y
        let mean_y = y.sum() / (n_samples as f64);
        self.initial_prediction_ = mean_y;

        let mut current_preds = Array1::<f64>::from_elem(n_samples, mean_y);
        let mut estimators = Vec::with_capacity(self.n_estimators);

        for m in 0..self.n_estimators {
            // Compute pseudo-residuals (negative gradient of squared error loss)
            let residuals = y - &current_preds;

            let seed = base_seed.wrapping_add(m as u64);
            let mut tree = DecisionTreeRegressor::new(
                self.max_depth,
                2,
                1,
                None, // Use all features
                Some(seed),
            );
            tree.categorical_features = self.categorical_features.clone();

            tree.fit(&X.view(), residuals.as_slice().unwrap());

            // Update predictions: Fm(x) = Fm_1(x) + lr * h_m(x)
            let tree_preds = tree.predict(X);
            for i in 0..n_samples {
                current_preds[i] += self.learning_rate * tree_preds[i];
            }

            estimators.push(tree);
        }

        self.estimators_ = estimators;
        Ok(())
    }

    pub fn predict(&self, X: &ArrayView2<f64>) -> Result<Array1<f64>, String> {
        if self.estimators_.is_empty() {
            return Err("Model is not fitted".to_string());
        }
        let n_samples = X.nrows();
        let mut preds = Array1::<f64>::from_elem(n_samples, self.initial_prediction_);

        for tree in &self.estimators_ {
            let tree_preds = tree.predict(X);
            for i in 0..n_samples {
                preds[i] += self.learning_rate * tree_preds[i];
            }
        }
        Ok(preds)
    }
}

// ==========================================
// GradientBoostingClassifier (Binary only for now, Log-Loss)
// ==========================================
pub struct GradientBoostingClassifier {
    pub n_estimators: usize,
    pub learning_rate: f64,
    pub max_depth: Option<usize>,
    pub random_state: Option<u64>,
    pub initial_prediction_: f64,
    pub classes_: Option<Vec<f64>>,
    pub estimators_: Vec<DecisionTreeRegressor>, // uses regressor to predict residuals
    pub categorical_features: Vec<usize>,
}

impl GradientBoostingClassifier {
    pub fn new(
        n_estimators: usize,
        learning_rate: f64,
        max_depth: Option<usize>,
        random_state: Option<u64>,
    ) -> Self {
        GradientBoostingClassifier {
            n_estimators,
            learning_rate,
            max_depth,
            random_state,
            initial_prediction_: 0.0,
            classes_: None,
            estimators_: Vec::new(),
            categorical_features: Vec::new(),
        }
    }

    #[inline]
    fn sigmoid(z: f64) -> f64 {
        1.0 / (1.0 + (-z).exp())
    }

    pub fn fit(&mut self, X: &ArrayView2<f64>, y: &ArrayView1<f64>) -> Result<(), String> {
        let n_samples = X.nrows();
        let base_seed = self.random_state.unwrap_or(0);

        let mut classes = y.to_vec();
        classes.sort_by(|a, b| a.partial_cmp(b).unwrap());
        classes.dedup();

        if classes.len() != 2 {
            return Err("GradientBoostingClassifier currently only supports binary classification".to_string());
        }

        let pos_class = classes[1];
        let mut y_bin = Array1::<f64>::zeros(n_samples);
        let mut sum_y = 0.0;
        for i in 0..n_samples {
            if y[i] == pos_class {
                y_bin[i] = 1.0;
                sum_y += 1.0;
            }
        }
        
        // F0 = log(p / (1-p))
        let p = sum_y / (n_samples as f64);
        let f0 = if p == 0.0 || p == 1.0 { 0.0 } else { (p / (1.0 - p)).ln() };
        self.initial_prediction_ = f0;

        let mut current_preds = Array1::<f64>::from_elem(n_samples, f0);
        let mut estimators = Vec::with_capacity(self.n_estimators);

        for m in 0..self.n_estimators {
            // Compute probabilities and residuals
            let mut residuals = Array1::<f64>::zeros(n_samples);
            for i in 0..n_samples {
                let prob = Self::sigmoid(current_preds[i]);
                residuals[i] = y_bin[i] - prob;
            }

            let seed = base_seed.wrapping_add(m as u64);
            let mut tree = DecisionTreeRegressor::new(
                self.max_depth,
                2,
                1,
                None, // Use all features
                Some(seed),
            );
            tree.categorical_features = self.categorical_features.clone();

            tree.fit(&X.view(), residuals.as_slice().unwrap());

            let tree_preds = tree.predict(X);
            for i in 0..n_samples {
                current_preds[i] += self.learning_rate * tree_preds[i];
            }

            estimators.push(tree);
        }

        self.classes_ = Some(classes);
        self.estimators_ = estimators;
        Ok(())
    }

    pub fn predict(&self, X: &ArrayView2<f64>) -> Result<Array1<f64>, String> {
        if self.estimators_.is_empty() {
            return Err("Model is not fitted".to_string());
        }
        let n_samples = X.nrows();
        let mut z = Array1::<f64>::from_elem(n_samples, self.initial_prediction_);

        for tree in &self.estimators_ {
            let tree_preds = tree.predict(X);
            for i in 0..n_samples {
                z[i] += self.learning_rate * tree_preds[i];
            }
        }

        let classes = self.classes_.as_ref().unwrap();
        let mut preds = Array1::<f64>::zeros(n_samples);
        for i in 0..n_samples {
            preds[i] = if Self::sigmoid(z[i]) >= 0.5 { classes[1] } else { classes[0] };
        }

        Ok(preds)
    }

    pub fn predict_proba(&self, X: &ArrayView2<f64>) -> Result<Array2<f64>, String> {
        if self.estimators_.is_empty() {
            return Err("Model is not fitted".to_string());
        }
        let n_samples = X.nrows();
        let mut z = Array1::<f64>::from_elem(n_samples, self.initial_prediction_);

        for tree in &self.estimators_ {
            let tree_preds = tree.predict(X);
            for i in 0..n_samples {
                z[i] += self.learning_rate * tree_preds[i];
            }
        }

        let mut proba = Array2::<f64>::zeros((n_samples, 2));
        for i in 0..n_samples {
            let p1 = Self::sigmoid(z[i]);
            proba[[i, 0]] = 1.0 - p1;
            proba[[i, 1]] = p1;
        }

        Ok(proba)
    }
}
