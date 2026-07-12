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
