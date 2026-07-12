
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

#[derive(serde::Serialize, serde::Deserialize)]
pub struct RFE {
    pub n_features_to_select: usize,
    pub step: usize,
    pub support_: Option<Vec<bool>>,
    pub ranking_: Option<Vec<usize>>,
}

impl RFE {
    pub fn new(n_features_to_select: usize, step: usize) -> Self {
        RFE {
            n_features_to_select,
            step,
            support_: None,
            ranking_: None,
        }
    }

    pub fn fit(&mut self, X: &ArrayView2<f64>, y: &ArrayView1<f64>) -> Result<(), String> {
        let n_features = X.ncols();
        let mut support = vec![true; n_features];
        let mut ranking = vec![1; n_features];
        
        let mut current_features = n_features;
        while current_features > self.n_features_to_select {
            let drop_count = self.step.min(current_features - self.n_features_to_select);
            let mut importance = vec![];
            for j in 0..n_features {
                if support[j] {
                    // dummy importance
                    importance.push((j, X[[0, j]].abs()));
                }
            }
            importance.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            
            for i in 0..drop_count {
                let idx = importance[i].0;
                support[idx] = false;
                ranking[idx] = current_features;
            }
            current_features -= drop_count;
        }
        
        self.support_ = Some(support);
        self.ranking_ = Some(ranking);
        Ok(())
    }

    pub fn transform(&self, X: &ArrayView2<f64>) -> Result<Array2<f64>, String> {
        let support = self.support_.as_ref().ok_or("Not fitted")?;
        let n_features = X.ncols();
        let selected_count = support.iter().filter(|&&x| x).count();
        let mut X_new = Array2::<f64>::zeros((X.nrows(), selected_count));
        
        let mut col_idx = 0;
        for j in 0..n_features {
            if support[j] {
                for i in 0..X.nrows() {
                    X_new[[i, col_idx]] = X[[i, j]];
                }
                col_idx += 1;
            }
        }
        Ok(X_new)
    }
}
