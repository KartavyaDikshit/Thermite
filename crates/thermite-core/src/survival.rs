use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rand::Rng;

#[derive(Clone)]
struct SurvivalNode {
    feature_idx: usize,
    threshold: f64,
    left: Option<Box<SurvivalNode>>,
    right: Option<Box<SurvivalNode>>,
    is_leaf: bool,
    n_samples: usize,
    /// Cumulative hazard estimates at each unique death time (Nelson-Aalen)
    cum_hazard: Vec<f64>,
    /// Unique death times for this leaf
    unique_times: Vec<f64>,
}

impl SurvivalNode {
    fn leaf(times: &[f64], events: &[f64]) -> Self {
        let mut unique_times: Vec<f64> = times.iter().copied().filter(|&t| t > 0.0).collect();
        unique_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        unique_times.dedup();

        let mut cum_hazard = Vec::with_capacity(unique_times.len());
        let mut cum = 0.0;
        for &t in &unique_times {
            let mut d = 0.0;
            let mut n_at_risk = 0.0;
            for i in 0..times.len() {
                if times[i] >= t {
                    n_at_risk += 1.0;
                    if times[i] == t && events[i] > 0.5 {
                        d += 1.0;
                    }
                }
            }
            if n_at_risk > 0.0 {
                cum += d / n_at_risk;
            }
            cum_hazard.push(cum);
        }

        SurvivalNode {
            is_leaf: true,
            left: None,
            right: None,
            feature_idx: 0,
            threshold: 0.0,
            unique_times,
            cum_hazard,
            n_samples: times.len(),
        }
    }
}

struct SurvivalTree {
    root: Option<Box<SurvivalNode>>,
    min_samples_leaf: usize,
    max_depth: usize,
}

impl SurvivalTree {
    fn new(min_samples_leaf: usize, max_depth: usize) -> Self {
        SurvivalTree { root: None, min_samples_leaf, max_depth }
    }

    fn fit(&mut self, X: &ArrayView2<f64>, times: &ArrayView1<f64>, events: &ArrayView1<f64>) {
        let indices: Vec<usize> = (0..X.nrows()).collect();
        self.root = Some(Box::new(Self::build_tree(X, times, events, &indices, self.min_samples_leaf, self.max_depth, 0)));
    }

    fn build_tree(X: &ArrayView2<f64>, times: &ArrayView1<f64>, events: &ArrayView1<f64>, indices: &[usize], min_samples_leaf: usize, max_depth: usize, depth: usize) -> SurvivalNode {
        let n = indices.len();
        if n <= min_samples_leaf || depth >= max_depth {
            let leaf_times: Vec<f64> = indices.iter().map(|&i| times[i]).collect();
            let leaf_events: Vec<f64> = indices.iter().map(|&i| events[i]).collect();
            return SurvivalNode::leaf(&leaf_times, &leaf_events);
        }

        let n_features = X.ncols();
        let mut best_score = -1.0;
        let mut best_feature = 0;
        let mut best_threshold = 0.0;

        for f in 0..n_features {
            let mut values: Vec<(usize, f64)> = indices.iter().map(|&i| (i, X[[i, f]])).collect();
            values.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            let mut n_left = 0;
            let mut n_right = values.len();
            let mut events_left = 0;

            for w in 0..values.len() - 1 {
                let i = values[w].0;
                n_left += 1;
                n_right -= 1;
                if events[i] > 0.5 {
                    events_left += 1;
                }

                if w + 1 < values.len() && (values[w + 1].1 - values[w].1).abs() < 1e-12 {
                    continue;
                }
                if n_left < min_samples_leaf || n_right < min_samples_leaf {
                    continue;
                }

                // Log-rank test statistic
                let mut num = 0.0;
                let mut den = 0.0;
                let n1 = n_left as f64;
                let n2 = n_right as f64;
                for k in 0..values.len() {
                    let idx = values[k].0;
                    if events[idx] > 0.5 {
                        let e1 = n1 / (n1 + n2);
                        num += events_left as f64 - e1;
                        den += e1 * (1.0 - e1) * (n1 + n2 - 1.0) / (n1 + n2);
                    }
                    if k == w { break; }
                }

                let score = if den > 0.0 { (num * num) / den } else { 0.0 };
                if score > best_score {
                    best_score = score;
                    best_feature = f;
                    best_threshold = (values[w].1 + values[w + 1].1) / 2.0;
                }
            }
        }

        if best_score < 0.0 || depth >= max_depth {
            let leaf_times: Vec<f64> = indices.iter().map(|&i| times[i]).collect();
            let leaf_events: Vec<f64> = indices.iter().map(|&i| events[i]).collect();
            return SurvivalNode::leaf(&leaf_times, &leaf_events);
        }

        let mut left_indices = Vec::new();
        let mut right_indices = Vec::new();
        for &i in indices {
            if X[[i, best_feature]] <= best_threshold {
                left_indices.push(i);
            } else {
                right_indices.push(i);
            }
        }

        SurvivalNode {
            is_leaf: false,
            left: Some(Box::new(Self::build_tree(X, times, events, &left_indices, min_samples_leaf, max_depth, depth + 1))),
            right: Some(Box::new(Self::build_tree(X, times, events, &right_indices, min_samples_leaf, max_depth, depth + 1))),
            feature_idx: best_feature,
            threshold: best_threshold,
            n_samples: indices.len(),
            cum_hazard: vec![],
            unique_times: vec![],
        }
    }

    fn predict_survival(&self, x: &[f64], times_to_predict: &[f64]) -> Vec<f64> {
        let mut node = self.root.as_ref().unwrap();
        while !node.is_leaf {
            if x[node.feature_idx] <= node.threshold {
                node = node.left.as_ref().unwrap();
            } else {
                node = node.right.as_ref().unwrap();
            }
        }
        let mut surv = Vec::with_capacity(times_to_predict.len());
        let mut j = 0;
        for &t in times_to_predict {
            while j < node.unique_times.len() && node.unique_times[j] <= t {
                j += 1;
            }
            let ch = if j > 0 { node.cum_hazard[j - 1] } else { 0.0 };
            surv.push((-ch).exp());
        }
        surv
    }
}

pub struct SurvivalForest {
    pub n_estimators: usize,
    pub max_depth: usize,
    pub min_samples_leaf: usize,
    pub fitted: bool,
    trees: Vec<SurvivalTree>,
}

impl SurvivalForest {
    pub fn new(n_estimators: usize) -> Self {
        SurvivalForest {
            n_estimators,
            max_depth: 10,
            min_samples_leaf: 5,
            fitted: false,
            trees: Vec::new(),
        }
    }

    pub fn fit(&mut self, X: &ArrayView2<f64>, times: &ArrayView1<f64>, events: &ArrayView1<f64>) -> Result<(), String> {
        let n_samples = X.nrows();
        let mut rng = rand::thread_rng();
        self.trees = (0..self.n_estimators).map(|_| {
            let mut bootstrap = Vec::with_capacity(n_samples);
            for _ in 0..n_samples {
                bootstrap.push(rng.gen_range(0..n_samples));
            }
            let mut tree = SurvivalTree::new(self.min_samples_leaf, self.max_depth);
            let boot_X = Array2::from_shape_fn((n_samples, X.ncols()), |(i, j)| X[[bootstrap[i], j]]);
            let boot_times = Array1::from_shape_fn(n_samples, |i| times[bootstrap[i]]);
            let boot_events = Array1::from_shape_fn(n_samples, |i| events[bootstrap[i]]);
            tree.fit(&boot_X.view(), &boot_times.view(), &boot_events.view());
            tree
        }).collect();
        self.fitted = true;
        Ok(())
    }

    pub fn predict_survival_function(&self, X: &ArrayView2<f64>, times_to_predict: &ArrayView1<f64>) -> Result<Array2<f64>, String> {
        if !self.fitted { return Err("Not fitted".to_string()); }
        let ttp: Vec<f64> = times_to_predict.iter().copied().collect();
        let mut result = Array2::zeros((X.nrows(), times_to_predict.len()));
        for i in 0..X.nrows() {
            let x_row: Vec<f64> = X.row(i).iter().copied().collect();
            let mut avg_surv = vec![0.0; times_to_predict.len()];
            for tree in &self.trees {
                let surv = tree.predict_survival(&x_row, &ttp);
                for j in 0..surv.len() {
                    avg_surv[j] += surv[j];
                }
            }
            for j in 0..avg_surv.len() {
                result[[i, j]] = avg_surv[j] / self.n_estimators as f64;
            }
        }
        Ok(result)
    }
}
