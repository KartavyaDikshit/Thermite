use ndarray::{Array1, Array2, Axis};

pub trait Estimator: Clone {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<(), String>;
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>, String>;
}

#[derive(Clone)]
pub struct MultiOutputRegressor<E: Estimator> {
    pub base_estimator: E,
    pub estimators: Vec<E>,
}

impl<E: Estimator> MultiOutputRegressor<E> {
    pub fn new(base_estimator: E) -> Self {
        Self {
            base_estimator,
            estimators: Vec::new(),
        }
    }

    pub fn fit(&mut self, x: &Array2<f64>, y: &Array2<f64>) -> Result<(), String> {
        let n_targets = y.ncols();
        self.estimators.clear();
        for i in 0..n_targets {
            let mut est = self.base_estimator.clone();
            let y_i = y.column(i).to_owned();
            est.fit(x, &y_i)?;
            self.estimators.push(est);
        }
        Ok(())
    }

    pub fn predict(&self, x: &Array2<f64>) -> Result<Array2<f64>, String> {
        if self.estimators.is_empty() {
            return Err("Estimator not fitted".to_string());
        }
        let n_samples = x.nrows();
        let n_targets = self.estimators.len();
        let mut predictions = Array2::<f64>::zeros((n_samples, n_targets));
        for (i, est) in self.estimators.iter().enumerate() {
            let pred_i = est.predict(x)?;
            predictions.column_mut(i).assign(&pred_i);
        }
        Ok(predictions)
    }
}
