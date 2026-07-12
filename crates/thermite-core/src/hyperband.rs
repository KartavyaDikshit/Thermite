use ndarray::{Array1, Array2};
use std::cmp::Ordering;

pub trait IncrementalEstimator: Clone {
    fn set_params(&mut self, params: &std::collections::HashMap<String, f64>);
    fn partial_fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<(), String>;
    fn score(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<f64, String>;
}

pub struct SuccessiveHalvingSearchCV<E: IncrementalEstimator> {
    pub base_estimator: E,
    pub param_grid: Vec<std::collections::HashMap<String, f64>>,
    pub min_resources: usize,
    pub factor: usize,
    pub best_estimator_: Option<E>,
    pub best_score_: f64,
}

impl<E: IncrementalEstimator> SuccessiveHalvingSearchCV<E> {
    pub fn new(base_estimator: E, param_grid: Vec<std::collections::HashMap<String, f64>>, min_resources: usize, factor: usize) -> Self {
        Self {
            base_estimator,
            param_grid,
            min_resources,
            factor,
            best_estimator_: None,
            best_score_: f64::NEG_INFINITY,
        }
    }

    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<(), String> {
        let n_samples = x.nrows();
        let mut active_estimators = Vec::new();

        // Initialize estimators with param combinations
        for params in &self.param_grid {
            let mut est = self.base_estimator.clone();
            est.set_params(params);
            active_estimators.push(est);
        }

        let mut current_resources = self.min_resources;

        while active_estimators.len() > 1 && current_resources <= n_samples {
            // Train all active estimators
            let x_slice = x.slice(ndarray::s![0..current_resources, ..]).to_owned();
            let y_slice = y.slice(ndarray::s![0..current_resources]).to_owned();

            for est in active_estimators.iter_mut() {
                est.partial_fit(&x_slice, &y_slice)?;
            }

            // Score active estimators
            let mut scores = Vec::new();
            for (idx, est) in active_estimators.iter().enumerate() {
                let score = est.score(&x_slice, &y_slice)?;
                scores.push((idx, score));
            }

            // Sort descending
            scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

            if active_estimators.len() == 1 {
                break; // handled by while condition
            }

            // Keep top (1/factor) fraction
            let keep_count = std::cmp::max(1, active_estimators.len() / self.factor);
            let mut next_estimators = Vec::new();
            for i in 0..keep_count {
                next_estimators.push(active_estimators[scores[i].0].clone());
            }

            active_estimators = next_estimators;
            current_resources *= self.factor;
        }

        // Final training and best estimator
        if !active_estimators.is_empty() {
            let mut best_est = active_estimators[0].clone();
            best_est.partial_fit(x, y)?;
            let final_score = best_est.score(x, y)?;
            self.best_estimator_ = Some(best_est);
            self.best_score_ = final_score;
        }

        Ok(())
    }
}
