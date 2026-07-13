use crate::linear_model::Ridge;
use ndarray::{Array1, ArrayView1, ArrayView2};

pub struct SurrogateOptimizer {
    model: Ridge,
}

impl SurrogateOptimizer {
    pub fn new(alpha: f64) -> Self {
        SurrogateOptimizer {
            model: Ridge::new(alpha, true),
        }
    }

    pub fn fit(&mut self, X: &ArrayView2<f64>, y: &ArrayView1<f64>) -> Result<(), String> {
        self.model.fit(X, y)
    }

    pub fn predict(&self, X: &ArrayView2<f64>) -> Result<Array1<f64>, String> {
        self.model.predict(X)
    }

    pub fn suggest_next(&self, X_candidates: &ArrayView2<f64>) -> Result<usize, String> {
        let preds = self.predict(X_candidates)?;
        let mut best_idx = 0;
        let mut best_score = preds[0];
        for i in 1..preds.len() {
            if preds[i] > best_score {
                best_score = preds[i];
                best_idx = i;
            }
        }
        Ok(best_idx)
    }
}
