use ndarray::{Array1, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};
use crate::linear_model::LinearRegression;

#[derive(Serialize, Deserialize)]
pub struct TLearner {
    pub model_control: LinearRegression,
    pub model_treatment: LinearRegression,
}

impl TLearner {
    pub fn new() -> Self {
        TLearner {
            model_control: LinearRegression::new(true),
            model_treatment: LinearRegression::new(true),
        }
    }

    pub fn fit(&mut self, x: &ArrayView2<f64>, treatment: &ArrayView1<f64>, y: &ArrayView1<f64>) -> Result<(), String> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        let mut x_control = Vec::new();
        let mut y_control = Vec::new();
        let mut x_treatment = Vec::new();
        let mut y_treatment = Vec::new();

        for i in 0..n_samples {
            let row = x.row(i).to_vec();
            if treatment[i] == 0.0 {
                x_control.extend_from_slice(&row);
                y_control.push(y[i]);
            } else {
                x_treatment.extend_from_slice(&row);
                y_treatment.push(y[i]);
            }
        }

        let x_c_arr = ndarray::Array2::from_shape_vec((y_control.len(), n_features), x_control)
            .map_err(|e| e.to_string())?;
        let y_c_arr = ndarray::Array1::from_vec(y_control);

        let x_t_arr = ndarray::Array2::from_shape_vec((y_treatment.len(), n_features), x_treatment)
            .map_err(|e| e.to_string())?;
        let y_t_arr = ndarray::Array1::from_vec(y_treatment);

        self.model_control.fit(&x_c_arr.view(), &y_c_arr.view())?;
        self.model_treatment.fit(&x_t_arr.view(), &y_t_arr.view())?;

        Ok(())
    }

    pub fn predict_cate(&self, x: &ArrayView2<f64>) -> Result<Array1<f64>, String> {
        let pred_control = self.model_control.predict(x)?;
        let pred_treatment = self.model_treatment.predict(x)?;

        Ok(pred_treatment - pred_control)
    }
}
