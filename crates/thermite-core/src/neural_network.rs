use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rand::prelude::*;
use thermite_gpu::{matmul, sigmoid, DeviceKind};

#[derive(serde::Serialize, serde::Deserialize, Clone)]
pub struct MLPClassifier {
    pub hidden_layer_sizes: Vec<usize>,
    pub learning_rate: f64,
    pub max_iter: usize,
    pub device: DeviceKind,
    pub weights: Vec<Vec<f32>>,
    pub biases: Vec<Vec<f32>>,
    pub classes_: Option<Vec<f64>>,
}

impl MLPClassifier {
    pub fn new(hidden_layer_sizes: Vec<usize>, learning_rate: f64, max_iter: usize, device: &str) -> Self {
        Self {
            hidden_layer_sizes,
            learning_rate,
            max_iter,
            device: DeviceKind::from_string(device),
            weights: Vec::new(),
            biases: Vec::new(),
            classes_: None,
        }
    }

    pub fn fit(&mut self, X: &ArrayView2<f64>, y: &ArrayView1<f64>) -> Result<(), String> {
        let n_samples = X.nrows();
        let n_features = X.ncols();
        
        let mut classes = y.to_vec();
        classes.sort_by(|a, b| a.partial_cmp(b).unwrap());
        classes.dedup();
        
        if classes.len() != 2 {
            return Err("MLPClassifier only supports binary classification for now".to_string());
        }
        
        self.classes_ = Some(classes.clone());
        let pos_class = classes[1];
        let mut y_bin = vec![0.0f32; n_samples];
        for i in 0..n_samples {
            if y[i] == pos_class {
                y_bin[i] = 1.0;
            }
        }
        
        let mut layer_sizes = vec![n_features];
        layer_sizes.extend(&self.hidden_layer_sizes);
        layer_sizes.push(1);
        
        let mut rng = rand::thread_rng();
        self.weights.clear();
        self.biases.clear();
        
        for i in 0..layer_sizes.len() - 1 {
            let in_size = layer_sizes[i];
            let out_size = layer_sizes[i+1];
            
            let mut w = vec![0.0f32; in_size * out_size];
            for j in 0..w.len() {
                w[j] = rng.gen_range(-0.1..0.1);
            }
            self.weights.push(w);
            
            let b = vec![0.0f32; out_size];
            self.biases.push(b);
        }
        
        let mut x_flat = vec![0.0f32; n_samples * n_features];
        for i in 0..n_samples {
            for j in 0..n_features {
                x_flat[i * n_features + j] = X[[i, j]] as f32;
            }
        }
        
        let lr = self.learning_rate as f32;
        
        for _iter in 0..self.max_iter {
            let mut activations = vec![x_flat.clone()];
            
            let mut current_a = x_flat.clone();
            let mut current_size = n_features;
            
            for (w, b) in self.weights.iter().zip(self.biases.iter()) {
                let out_size = b.len();
                let z = matmul(&current_a, w, n_samples, current_size, out_size, self.device);
                
                let mut z_biased = z;
                for i in 0..n_samples {
                    for j in 0..out_size {
                        z_biased[i * out_size + j] += b[j];
                    }
                }
                
                current_a = sigmoid(&z_biased, self.device);
                activations.push(current_a.clone());
                current_size = out_size;
            }
            
            let out_a = activations.last().unwrap();
            let mut delta = vec![0.0f32; n_samples];
            for i in 0..n_samples {
                delta[i] = (out_a[i] - y_bin[i]) / n_samples as f32;
            }
            
            for l in (0..self.weights.len()).rev() {
                let a_prev = &activations[l];
                let in_size = self.weights[l].len() / self.biases[l].len();
                let out_size = self.biases[l].len();
                
                let mut a_prev_t = vec![0.0f32; in_size * n_samples];
                for i in 0..n_samples {
                    for j in 0..in_size {
                        a_prev_t[j * n_samples + i] = a_prev[i * in_size + j];
                    }
                }
                
                let dW = matmul(&a_prev_t, &delta, in_size, n_samples, out_size, self.device);
                
                let mut db = vec![0.0f32; out_size];
                for i in 0..n_samples {
                    for j in 0..out_size {
                        db[j] += delta[i * out_size + j];
                    }
                }
                
                if l > 0 {
                    let w = &self.weights[l];
                    let mut w_t = vec![0.0f32; out_size * in_size];
                    for i in 0..in_size {
                        for j in 0..out_size {
                            w_t[j * in_size + i] = w[i * out_size + j];
                        }
                    }
                    let mut next_delta = matmul(&delta, &w_t, n_samples, out_size, in_size, self.device);
                    
                    for i in 0..n_samples * in_size {
                        let a = a_prev[i];
                        next_delta[i] *= a * (1.0 - a);
                    }
                    delta = next_delta;
                }
                
                for i in 0..self.weights[l].len() {
                    self.weights[l][i] -= lr * dW[i];
                }
                for i in 0..self.biases[l].len() {
                    self.biases[l][i] -= lr * db[i];
                }
            }
        }
        
        Ok(())
    }

    pub fn predict_proba(&self, X: &ArrayView2<f64>) -> Result<Array2<f64>, String> {
        if self.classes_.is_none() {
            return Err("Model not fitted".to_string());
        }
        let n_samples = X.nrows();
        let n_features = X.ncols();
        
        let mut current_a = vec![0.0f32; n_samples * n_features];
        for i in 0..n_samples {
            for j in 0..n_features {
                current_a[i * n_features + j] = X[[i, j]] as f32;
            }
        }
        
        let mut current_size = n_features;
        
        for (w, b) in self.weights.iter().zip(self.biases.iter()) {
            let out_size = b.len();
            let z = matmul(&current_a, w, n_samples, current_size, out_size, self.device);
            
            let mut z_biased = z;
            for i in 0..n_samples {
                for j in 0..out_size {
                    z_biased[i * out_size + j] += b[j];
                }
            }
            
            current_a = sigmoid(&z_biased, self.device);
            current_size = out_size;
        }
        
        let mut proba = Array2::<f64>::zeros((n_samples, 2));
        for i in 0..n_samples {
            let p1 = current_a[i] as f64;
            proba[[i, 0]] = 1.0 - p1;
            proba[[i, 1]] = p1;
        }
        
        Ok(proba)
    }

    pub fn predict(&self, X: &ArrayView2<f64>) -> Result<Array1<f64>, String> {
        let proba = self.predict_proba(X)?;
        let classes = self.classes_.as_ref().unwrap();
        let mut preds = Array1::<f64>::zeros(X.nrows());
        for i in 0..X.nrows() {
            if proba[[i, 1]] >= 0.5 {
                preds[i] = classes[1];
            } else {
                preds[i] = classes[0];
            }
        }
        Ok(preds)
    }
}
