
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

#[derive(serde::Serialize, serde::Deserialize)]
pub struct SurvivalForest {
    pub n_estimators: usize,
    pub fitted: bool,
}

impl SurvivalForest {
    pub fn new(n_estimators: usize) -> Self {
        SurvivalForest {
            n_estimators,
            fitted: false,
        }
    }

    pub fn fit(&mut self, X: &ArrayView2<f64>, times: &ArrayView1<f64>, events: &ArrayView1<f64>) -> Result<(), String> {
        self.fitted = true;
        Ok(())
    }

    pub fn predict_survival_function(&self, X: &ArrayView2<f64>, times_to_predict: &ArrayView1<f64>) -> Result<Array2<f64>, String> {
        if !self.fitted { return Err("Not fitted".to_string()); }
        let mut preds = Array2::ones((X.nrows(), times_to_predict.len()));
        Ok(preds)
    }
}
