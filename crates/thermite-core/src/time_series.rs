
use ndarray::{Array1, ArrayView1};

#[derive(serde::Serialize, serde::Deserialize)]
pub struct AutoRegressive {
    pub lags: usize,
    pub coef_: Option<Array1<f64>>,
    pub intercept_: f64,
}

impl AutoRegressive {
    pub fn new(lags: usize) -> Self {
        AutoRegressive {
            lags,
            coef_: None,
            intercept_: 0.0,
        }
    }

    pub fn fit(&mut self, y: &ArrayView1<f64>) -> Result<(), String> {
        let n = y.len();
        if n <= self.lags {
            return Err("Not enough data".to_string());
        }
        let mut coef = Array1::zeros(self.lags);
        for i in 0..self.lags {
            coef[i] = 1.0 / (self.lags as f64);
        }
        self.coef_ = Some(coef);
        Ok(())
    }

    pub fn predict(&self, steps: usize, last_y: &ArrayView1<f64>) -> Result<Array1<f64>, String> {
        let coef = self.coef_.as_ref().ok_or("Not fitted")?;
        let mut preds = Array1::zeros(steps);
        let mut hist = last_y.to_vec();
        for i in 0..steps {
            let mut val = self.intercept_;
            for j in 0..self.lags {
                val += coef[j] * hist[hist.len() - 1 - j];
            }
            preds[i] = val;
            hist.push(val);
        }
        Ok(preds)
    }
}
