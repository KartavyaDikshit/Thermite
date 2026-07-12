#![allow(non_snake_case)]
use ndarray::{Array2, ArrayView2, Axis};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

// ==========================================
// Tree Node
// ==========================================

/// A single node in the decision tree, stored in a flat Vec.
/// Internal nodes have `left` and `right` as indices into the Vec.
/// Leaf nodes have `left == usize::MAX` and `right == usize::MAX`.
#[derive(Debug, Clone)]
pub struct TreeNode {
    pub feature_idx: usize,
    pub threshold: f64,
    pub is_categorical: bool,
    pub left_categories: Vec<f64>,
    pub left: usize,
    pub right: usize,
    /// For classifier leaves: class probabilities (length = n_classes).
    /// For regressor leaves: single-element vec with the predicted value.
    pub value: Vec<f64>,
    pub n_samples: usize,
}

impl TreeNode {
    fn leaf(value: Vec<f64>, n_samples: usize) -> Self {
        TreeNode {
            feature_idx: 0,
            threshold: 0.0,
            is_categorical: false,
            left_categories: Vec::new(),
            left: usize::MAX,
            right: usize::MAX,
            value,
            n_samples,
        }
    }

    pub fn is_leaf(&self) -> bool {
        self.left == usize::MAX
    }
}

// ==========================================
// Splitting criteria
// ==========================================

/// Gini impurity for a set of class counts.
fn gini_impurity(class_counts: &[f64], total: f64) -> f64 {
    if total == 0.0 {
        return 0.0;
    }
    let mut sum_sq = 0.0;
    for &c in class_counts {
        let p = c / total;
        sum_sq += p * p;
    }
    1.0 - sum_sq
}

/// MSE for a set of values given their sum and sum-of-squares.
fn mse_from_stats(sum: f64, sum_sq: f64, count: f64) -> f64 {
    if count == 0.0 {
        return 0.0;
    }
    let mean = sum / count;
    sum_sq / count - mean * mean
}

// ==========================================
// Shared split-finding helpers
// ==========================================

/// Candidate feature indices, with optional random subsampling.
fn select_features(n_features: usize, max_features: Option<usize>, rng: &mut SmallRng) -> Vec<usize> {
    let mut features: Vec<usize> = (0..n_features).collect();
    if let Some(mf) = max_features {
        if mf < n_features {
            features.shuffle(rng);
            features.truncate(mf);
        }
    }
    features
}

/// Sort sample indices by feature value (ascending).
fn sorted_indices_by_feature(X: &ArrayView2<f64>, indices: &[usize], feature: usize) -> Vec<usize> {
    let mut sorted = indices.to_vec();
    sorted.sort_unstable_by(|&a, &b| {
        X[[a, feature]].partial_cmp(&X[[b, feature]]).unwrap_or(std::cmp::Ordering::Equal)
    });
    sorted
}

// ==========================================
// DecisionTreeClassifier
// ==========================================

pub struct DecisionTreeClassifier {
    pub max_depth: Option<usize>,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    pub max_features: Option<usize>,
    pub random_state: Option<u64>,
    pub nodes: Vec<TreeNode>,
    pub n_classes: usize,
    pub classes: Vec<f64>,
    pub categorical_features: Vec<usize>,
}

impl DecisionTreeClassifier {
    pub fn new(
        max_depth: Option<usize>,
        min_samples_split: usize,
        min_samples_leaf: usize,
        max_features: Option<usize>,
        random_state: Option<u64>,
    ) -> Self {
        DecisionTreeClassifier {
            max_depth,
            min_samples_split: min_samples_split.max(2),
            min_samples_leaf: min_samples_leaf.max(1),
            max_features,
            random_state,
            nodes: Vec::new(),
            n_classes: 0,
            classes: Vec::new(),
            categorical_features: Vec::new(),
        }
    }

    pub fn fit(&mut self, X: &ArrayView2<f64>, y: &[f64]) {
        let n_samples = X.nrows();
        assert_eq!(n_samples, y.len(), "X and y must have the same number of samples");
        assert!(n_samples > 0, "Cannot fit on empty data");

        // Discover classes (sorted unique values)
        let mut classes: Vec<f64> = y.to_vec();
        classes.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        classes.dedup();
        self.n_classes = classes.len();
        self.classes = classes;

        let mut rng = match self.random_state {
            Some(seed) => SmallRng::seed_from_u64(seed),
            None => SmallRng::from_entropy(),
        };

        let indices: Vec<usize> = (0..n_samples).collect();
        self.nodes.clear();
        self.build_classifier_tree(X, y, &indices, 0, &mut rng);
    }

    fn build_classifier_tree(
        &mut self,
        X: &ArrayView2<f64>,
        y: &[f64],
        indices: &[usize],
        depth: usize,
        rng: &mut SmallRng,
    ) -> usize {
        let n = indices.len();
        let n_features = X.ncols();

        // Compute class counts for current node
        let class_counts = self.compute_class_counts(y, indices);
        let total = n as f64;

        // Check stopping conditions
        let is_pure = class_counts.iter().any(|&c| (c - total).abs() < 1e-12);
        let at_max_depth = self.max_depth.map_or(false, |md| depth >= md);
        let too_few_samples = n < self.min_samples_split;

        if is_pure || at_max_depth || too_few_samples {
            // Make leaf node with class probabilities
            let probs: Vec<f64> = class_counts.iter().map(|&c| c / total).collect();
            let node_idx = self.nodes.len();
            self.nodes.push(TreeNode::leaf(probs, n));
            return node_idx;
        }

        // Find best split
        let candidate_features = select_features(n_features, self.max_features, rng);
        let best_split = self.find_best_classification_split(
            X, y, indices, &candidate_features, &class_counts, total,
        );

        match best_split {
            None => {
                let probs: Vec<f64> = class_counts.iter().map(|&c| c / total).collect();
                let node_idx = self.nodes.len();
                self.nodes.push(TreeNode::leaf(probs, n));
                node_idx
            }
            Some((best_feature, best_threshold, is_categorical, left_categories, left_indices, right_indices)) => {
                let node_idx = self.nodes.len();
                self.nodes.push(TreeNode::leaf(vec![], n));

                let left_child = self.build_classifier_tree(X, y, &left_indices, depth + 1, rng);
                let right_child = self.build_classifier_tree(X, y, &right_indices, depth + 1, rng);

                let probs: Vec<f64> = class_counts.iter().map(|&c| c / total).collect();
                self.nodes[node_idx] = TreeNode {
                    feature_idx: best_feature,
                    threshold: best_threshold,
                    is_categorical,
                    left_categories,
                    left: left_child,
                    right: right_child,
                    value: probs,
                    n_samples: n,
                };
                node_idx
            }
        }
    }

    fn compute_class_counts(&self, y: &[f64], indices: &[usize]) -> Vec<f64> {
        let mut counts = vec![0.0; self.n_classes];
        for &i in indices {
            let class_idx = self.classes
                .iter()
                .position(|&c| (c - y[i]).abs() < 1e-12)
                .unwrap();
            counts[class_idx] += 1.0;
        }
        counts
    }

    fn find_best_classification_split(
        &self,
        X: &ArrayView2<f64>,
        y: &[f64],
        indices: &[usize],
        candidate_features: &[usize],
        parent_counts: &[f64],
        total: f64,
    ) -> Option<(usize, f64, bool, Vec<f64>, Vec<usize>, Vec<usize>)> {
        let parent_impurity = gini_impurity(parent_counts, total);
        let mut best_gain = 0.0;
        let mut best_split: Option<(usize, f64, bool, Vec<f64>, Vec<usize>, Vec<usize>)> = None;

        for &feat in candidate_features {
            if self.categorical_features.contains(&feat) {
                let mut cat_counts = std::collections::HashMap::new();
                for &idx in indices {
                    let val = X[[idx, feat]];
                    let c = self.classes.iter().position(|&c| (c - y[idx]).abs() < 1e-12).unwrap();
                    let entry = cat_counts.entry(val.to_bits()).or_insert((0.0, vec![0.0; self.n_classes]));
                    entry.0 += 1.0;
                    entry.1[c] += 1.0;
                }
                if cat_counts.len() <= 1 {
                    continue;
                }

                let mut maj_class = 0;
                let mut max_count = parent_counts[0];
                for i in 1..self.n_classes {
                    if parent_counts[i] > max_count {
                        max_count = parent_counts[i];
                        maj_class = i;
                    }
                }

                let mut cats: Vec<(f64, f64, f64, Vec<f64>)> = cat_counts.into_iter().map(|(k, v)| {
                    let val = f64::from_bits(k);
                    let prob = v.1[maj_class] / v.0;
                    (val, prob, v.0, v.1)
                }).collect();
                cats.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

                let mut left_counts = vec![0.0; self.n_classes];
                let mut right_counts = parent_counts.to_vec();
                let mut left_total = 0.0;
                let mut right_total = total;
                let mut current_left_cats = Vec::new();

                for i in 0..cats.len() - 1 {
                    let cat = &cats[i];
                    current_left_cats.push(cat.0);
                    left_total += cat.2;
                    right_total -= cat.2;
                    for c in 0..self.n_classes {
                        left_counts[c] += cat.3[c];
                        right_counts[c] -= cat.3[c];
                    }

                    if (left_total as usize) < self.min_samples_leaf || (right_total as usize) < self.min_samples_leaf {
                        continue;
                    }

                    let left_impurity = gini_impurity(&left_counts, left_total);
                    let right_impurity = gini_impurity(&right_counts, right_total);
                    let weighted = (left_total * left_impurity + right_total * right_impurity) / total;
                    let gain = parent_impurity - weighted;

                    if gain > best_gain {
                        best_gain = gain;
                        let mut left_idx = Vec::with_capacity(left_total as usize);
                        let mut right_idx = Vec::with_capacity(right_total as usize);
                        for &idx in indices {
                            if current_left_cats.contains(&X[[idx, feat]]) {
                                left_idx.push(idx);
                            } else {
                                right_idx.push(idx);
                            }
                        }
                        best_split = Some((feat, 0.0, true, current_left_cats.clone(), left_idx, right_idx));
                    }
                }
            } else {
                let sorted = sorted_indices_by_feature(X, indices, feat);
                let mut left_counts = vec![0.0; self.n_classes];
                let mut right_counts = parent_counts.to_vec();
            let mut left_total = 0.0;
            let mut right_total = total;

            for i in 0..sorted.len() - 1 {
                let idx = sorted[i];
                let class_idx = self.classes
                    .iter()
                    .position(|&c| (c - y[idx]).abs() < 1e-12)
                    .unwrap();
                left_counts[class_idx] += 1.0;
                right_counts[class_idx] -= 1.0;
                left_total += 1.0;
                right_total -= 1.0;

                // Skip if consecutive values are the same
                let val = X[[idx, feat]];
                let next_val = X[[sorted[i + 1], feat]];
                if (val - next_val).abs() < 1e-12 {
                    continue;
                }

                // Check min_samples_leaf
                if (left_total as usize) < self.min_samples_leaf
                    || (right_total as usize) < self.min_samples_leaf
                {
                    continue;
                }

                let left_impurity = gini_impurity(&left_counts, left_total);
                let right_impurity = gini_impurity(&right_counts, right_total);
                let weighted = (left_total * left_impurity + right_total * right_impurity) / total;
                let gain = parent_impurity - weighted;

                if gain > best_gain {
                    best_gain = gain;
                    let threshold = (val + next_val) / 2.0;
                    let left_idx: Vec<usize> = sorted[..=i].to_vec();
                    let right_idx: Vec<usize> = sorted[i + 1..].to_vec();
                    best_split = Some((feat, threshold, false, vec![], left_idx, right_idx));
                }
            }
        }
        }

        best_split
    }

    pub fn predict(&self, X: &ArrayView2<f64>) -> Vec<f64> {
        assert!(!self.nodes.is_empty(), "Tree is not fitted");
        (0..X.nrows())
            .map(|i| {
                let probs = self.predict_proba_single(X, i);
                // Return class with highest probability
                let mut best_idx = 0;
                let mut best_prob = probs[0];
                for (j, &p) in probs.iter().enumerate().skip(1) {
                    if p > best_prob {
                        best_prob = p;
                        best_idx = j;
                    }
                }
                self.classes[best_idx]
            })
            .collect()
    }

    pub fn predict_proba(&self, X: &ArrayView2<f64>) -> Array2<f64> {
        assert!(!self.nodes.is_empty(), "Tree is not fitted");
        let n_samples = X.nrows();
        let mut result = Array2::zeros((n_samples, self.n_classes));
        for i in 0..n_samples {
            let probs = self.predict_proba_single(X, i);
            for (j, &p) in probs.iter().enumerate() {
                result[[i, j]] = p;
            }
        }
        result
    }

    fn predict_proba_single(&self, X: &ArrayView2<f64>, sample_idx: usize) -> &[f64] {
        let mut node_idx = 0;
        loop {
            let node = &self.nodes[node_idx];
            if node.is_leaf() {
                return &node.value;
            }
            if node.is_categorical {
                if node.left_categories.contains(&X[[sample_idx, node.feature_idx]]) {
                    node_idx = node.left;
                } else {
                    node_idx = node.right;
                }
            } else {
                if X[[sample_idx, node.feature_idx]] <= node.threshold {
                    node_idx = node.left;
                } else {
                    node_idx = node.right;
                }
            }
        }
    }
}

// ==========================================
// DecisionTreeRegressor
// ==========================================

pub struct DecisionTreeRegressor {
    pub max_depth: Option<usize>,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    pub max_features: Option<usize>,
    pub random_state: Option<u64>,
    pub nodes: Vec<TreeNode>,
    pub categorical_features: Vec<usize>,
}

impl DecisionTreeRegressor {
    pub fn new(
        max_depth: Option<usize>,
        min_samples_split: usize,
        min_samples_leaf: usize,
        max_features: Option<usize>,
        random_state: Option<u64>,
    ) -> Self {
        DecisionTreeRegressor {
            max_depth,
            min_samples_split: min_samples_split.max(2),
            min_samples_leaf: min_samples_leaf.max(1),
            max_features,
            random_state,
            nodes: Vec::new(),
            categorical_features: Vec::new(),
        }
    }

    pub fn fit(&mut self, X: &ArrayView2<f64>, y: &[f64]) {
        let n_samples = X.nrows();
        assert_eq!(n_samples, y.len(), "X and y must have the same number of samples");
        assert!(n_samples > 0, "Cannot fit on empty data");

        let mut rng = match self.random_state {
            Some(seed) => SmallRng::seed_from_u64(seed),
            None => SmallRng::from_entropy(),
        };

        let indices: Vec<usize> = (0..n_samples).collect();
        self.nodes.clear();
        self.build_regressor_tree(X, y, &indices, 0, &mut rng);
    }

    fn build_regressor_tree(
        &mut self,
        X: &ArrayView2<f64>,
        y: &[f64],
        indices: &[usize],
        depth: usize,
        rng: &mut SmallRng,
    ) -> usize {
        let n = indices.len();
        let n_features = X.ncols();

        // Compute mean and statistics
        let sum: f64 = indices.iter().map(|&i| y[i]).sum();
        let sum_sq: f64 = indices.iter().map(|&i| y[i] * y[i]).sum();
        let mean = sum / n as f64;

        // Check stopping conditions
        let at_max_depth = self.max_depth.map_or(false, |md| depth >= md);
        let too_few_samples = n < self.min_samples_split;
        let is_constant = {
            let first = y[indices[0]];
            indices.iter().all(|&i| (y[i] - first).abs() < 1e-12)
        };

        if at_max_depth || too_few_samples || is_constant {
            let node_idx = self.nodes.len();
            self.nodes.push(TreeNode::leaf(vec![mean], n));
            return node_idx;
        }

        // Find best split
        let candidate_features = select_features(n_features, self.max_features, rng);
        let best_split = self.find_best_regression_split(
            X, y, indices, &candidate_features, sum, sum_sq, n as f64,
        );

        match best_split {
            None => {
                let node_idx = self.nodes.len();
                self.nodes.push(TreeNode::leaf(vec![mean], n));
                node_idx
            }
            Some((best_feature, best_threshold, is_categorical, left_categories, left_indices, right_indices)) => {
                let node_idx = self.nodes.len();
                self.nodes.push(TreeNode::leaf(vec![], n));

                let left_child = self.build_regressor_tree(X, y, &left_indices, depth + 1, rng);
                let right_child = self.build_regressor_tree(X, y, &right_indices, depth + 1, rng);

                self.nodes[node_idx] = TreeNode {
                    feature_idx: best_feature,
                    threshold: best_threshold,
                    is_categorical,
                    left_categories,
                    left: left_child,
                    right: right_child,
                    value: vec![mean],
                    n_samples: n,
                };
                node_idx
            }
        }
    }

    fn find_best_regression_split(
        &self,
        X: &ArrayView2<f64>,
        y: &[f64],
        indices: &[usize],
        candidate_features: &[usize],
        total_sum: f64,
        total_sum_sq: f64,
        total_count: f64,
    ) -> Option<(usize, f64, bool, Vec<f64>, Vec<usize>, Vec<usize>)> {
        let parent_mse = mse_from_stats(total_sum, total_sum_sq, total_count);
        let mut best_gain = 0.0;
        let mut best_split: Option<(usize, f64, bool, Vec<f64>, Vec<usize>, Vec<usize>)> = None;

        for &feat in candidate_features {
            if self.categorical_features.contains(&feat) {
                let mut cat_stats = std::collections::HashMap::new();
                for &idx in indices {
                    let val = X[[idx, feat]];
                    let target = y[idx];
                    let entry = cat_stats.entry(val.to_bits()).or_insert((0.0, 0.0, 0.0));
                    entry.0 += 1.0;
                    entry.1 += target;
                    entry.2 += target * target;
                }
                if cat_stats.len() <= 1 {
                    continue;
                }

                let mut cats: Vec<(f64, f64, f64, f64, f64)> = cat_stats.into_iter().map(|(k, v)| {
                    let val = f64::from_bits(k);
                    let mean = v.1 / v.0;
                    (val, mean, v.0, v.1, v.2)
                }).collect();
                cats.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

                let mut left_sum = 0.0;
                let mut left_sum_sq = 0.0;
                let mut left_count = 0.0;
                let mut current_left_cats = Vec::new();

                for i in 0..cats.len() - 1 {
                    let cat = &cats[i];
                    current_left_cats.push(cat.0);
                    left_count += cat.2;
                    left_sum += cat.3;
                    left_sum_sq += cat.4;

                    let right_count = total_count - left_count;
                    let right_sum = total_sum - left_sum;
                    let right_sum_sq = total_sum_sq - left_sum_sq;

                    if (left_count as usize) < self.min_samples_leaf || (right_count as usize) < self.min_samples_leaf {
                        continue;
                    }

                    let left_mse = mse_from_stats(left_sum, left_sum_sq, left_count);
                    let right_mse = mse_from_stats(right_sum, right_sum_sq, right_count);
                    let weighted = (left_count * left_mse + right_count * right_mse) / total_count;
                    let gain = parent_mse - weighted;

                    if gain > best_gain {
                        best_gain = gain;
                        let mut left_idx = Vec::with_capacity(left_count as usize);
                        let mut right_idx = Vec::with_capacity(right_count as usize);
                        for &idx in indices {
                            if current_left_cats.contains(&X[[idx, feat]]) {
                                left_idx.push(idx);
                            } else {
                                right_idx.push(idx);
                            }
                        }
                        best_split = Some((feat, 0.0, true, current_left_cats.clone(), left_idx, right_idx));
                    }
                }
            } else {
                let sorted = sorted_indices_by_feature(X, indices, feat);
                let mut left_sum = 0.0;
                let mut left_sum_sq = 0.0;
                let mut left_count = 0.0;

            for i in 0..sorted.len() - 1 {
                let idx = sorted[i];
                let val_y = y[idx];
                left_sum += val_y;
                left_sum_sq += val_y * val_y;
                left_count += 1.0;

                let right_sum = total_sum - left_sum;
                let right_sum_sq = total_sum_sq - left_sum_sq;
                let right_count = total_count - left_count;

                // Skip if consecutive values are the same
                let val = X[[idx, feat]];
                let next_val = X[[sorted[i + 1], feat]];
                if (val - next_val).abs() < 1e-12 {
                    continue;
                }

                // Check min_samples_leaf
                if (left_count as usize) < self.min_samples_leaf
                    || (right_count as usize) < self.min_samples_leaf
                {
                    continue;
                }

                let left_mse = mse_from_stats(left_sum, left_sum_sq, left_count);
                let right_mse = mse_from_stats(right_sum, right_sum_sq, right_count);
                let weighted = (left_count * left_mse + right_count * right_mse) / total_count;
                let gain = parent_mse - weighted;

                if gain > best_gain {
                    best_gain = gain;
                    let threshold = (val + next_val) / 2.0;
                    let left_idx: Vec<usize> = sorted[..=i].to_vec();
                    let right_idx: Vec<usize> = sorted[i + 1..].to_vec();
                    best_split = Some((feat, threshold, false, vec![], left_idx, right_idx));
                }
            }
            }
        }

        best_split
    }

    pub fn predict(&self, X: &ArrayView2<f64>) -> Vec<f64> {
        assert!(!self.nodes.is_empty(), "Tree is not fitted");
        (0..X.nrows())
            .map(|i| self.predict_single(X, i))
            .collect()
    }

    fn predict_single(&self, X: &ArrayView2<f64>, sample_idx: usize) -> f64 {
        let mut node_idx = 0;
        loop {
            let node = &self.nodes[node_idx];
            if node.is_leaf() {
                return node.value[0];
            }
            if node.is_categorical {
                if node.left_categories.contains(&X[[sample_idx, node.feature_idx]]) {
                    node_idx = node.left;
                } else {
                    node_idx = node.right;
                }
            } else {
                if X[[sample_idx, node.feature_idx]] <= node.threshold {
                    node_idx = node.left;
                } else {
                    node_idx = node.right;
                }
            }
        }
    }
}

// ==========================================
// Helper: predict a single sample through a tree's node vec
// ==========================================

/// Traverse the tree for a single sample, returning the leaf node's value slice.
pub fn traverse_tree<'a>(nodes: &'a [TreeNode], X: &ArrayView2<f64>, sample_idx: usize) -> &'a [f64] {
    let mut node_idx = 0;
    loop {
        let node = &nodes[node_idx];
        if node.is_leaf() {
            return &node.value;
        }
        if node.is_categorical {
            if node.left_categories.contains(&X[[sample_idx, node.feature_idx]]) {
                node_idx = node.left;
            } else {
                node_idx = node.right;
            }
        } else {
            if X[[sample_idx, node.feature_idx]] <= node.threshold {
                node_idx = node.left;
            } else {
                node_idx = node.right;
            }
        }
    }
}

// ==========================================
// Tests
// ==========================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_gini_impurity_pure() {
        let counts = vec![10.0, 0.0];
        assert!((gini_impurity(&counts, 10.0) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_gini_impurity_balanced() {
        let counts = vec![5.0, 5.0];
        assert!((gini_impurity(&counts, 10.0) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_mse_from_stats_constant() {
        // All values are 3.0, count=4
        let sum = 12.0;
        let sum_sq = 36.0;
        assert!((mse_from_stats(sum, sum_sq, 4.0) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_mse_from_stats_variance() {
        // values: [1, 2, 3, 4], mean=2.5, mse = ((1-2.5)^2 + ... )/4 = 1.25
        let sum = 10.0;
        let sum_sq = 30.0;
        assert!((mse_from_stats(sum, sum_sq, 4.0) - 1.25).abs() < 1e-12);
    }

    // ---- DecisionTreeClassifier tests ----

    #[test]
    fn test_classifier_pure_data() {
        let X = array![[1.0], [2.0], [3.0]];
        let y = vec![0.0, 0.0, 0.0];
        let mut tree = DecisionTreeClassifier::new(None, 2, 1, None, Some(42));
        tree.fit(&X.view(), &y);

        let preds = tree.predict(&X.view());
        assert_eq!(preds, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_classifier_simple_split() {
        // Simple linearly separable data
        let X = array![
            [1.0], [2.0], [3.0], [4.0],
            [5.0], [6.0], [7.0], [8.0]
        ];
        let y = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let mut tree = DecisionTreeClassifier::new(None, 2, 1, None, Some(42));
        tree.fit(&X.view(), &y);

        let preds = tree.predict(&X.view());
        assert_eq!(preds, y);
    }

    #[test]
    fn test_classifier_predict_proba() {
        let X = array![
            [1.0], [2.0], [3.0], [4.0],
            [5.0], [6.0], [7.0], [8.0]
        ];
        let y = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let mut tree = DecisionTreeClassifier::new(None, 2, 1, None, Some(42));
        tree.fit(&X.view(), &y);

        let proba = tree.predict_proba(&X.view());
        assert_eq!(proba.nrows(), 8);
        assert_eq!(proba.ncols(), 2);
        // First sample should have high probability for class 0
        assert!(proba[[0, 0]] > 0.9);
        // Last sample should have high probability for class 1
        assert!(proba[[7, 1]] > 0.9);
    }

    #[test]
    fn test_classifier_max_depth() {
        let X = array![
            [1.0, 1.0], [2.0, 1.0], [3.0, 1.0],
            [1.0, 2.0], [2.0, 2.0], [3.0, 2.0]
        ];
        let y = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0];
        let mut tree = DecisionTreeClassifier::new(Some(1), 2, 1, None, Some(42));
        tree.fit(&X.view(), &y);

        // With max_depth=1, the root + 2 leaf nodes
        assert!(tree.nodes.len() <= 3);
    }

    #[test]
    fn test_classifier_min_samples_leaf() {
        let X = array![
            [1.0], [2.0], [3.0], [4.0], [5.0], [6.0]
        ];
        let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let mut tree = DecisionTreeClassifier::new(None, 2, 3, None, Some(42));
        tree.fit(&X.view(), &y);

        // Each leaf must have at least 3 samples, so with 6 samples we get 1 split
        let leaf_counts: Vec<usize> = tree.nodes.iter().filter(|n| n.is_leaf()).map(|n| n.n_samples).collect();
        for &count in &leaf_counts {
            assert!(count >= 3);
        }
    }

    #[test]
    fn test_classifier_multiclass() {
        let X = array![
            [1.0], [2.0], [3.0],
            [4.0], [5.0], [6.0],
            [7.0], [8.0], [9.0]
        ];
        let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0];
        let mut tree = DecisionTreeClassifier::new(None, 2, 1, None, Some(42));
        tree.fit(&X.view(), &y);

        let preds = tree.predict(&X.view());
        assert_eq!(preds, y);
        assert_eq!(tree.n_classes, 3);

        let proba = tree.predict_proba(&X.view());
        assert_eq!(proba.ncols(), 3);
    }

    // ---- DecisionTreeRegressor tests ----

    #[test]
    fn test_regressor_constant_target() {
        let X = array![[1.0], [2.0], [3.0]];
        let y = vec![5.0, 5.0, 5.0];
        let mut tree = DecisionTreeRegressor::new(None, 2, 1, None, Some(42));
        tree.fit(&X.view(), &y);

        let preds = tree.predict(&X.view());
        for &p in &preds {
            assert!((p - 5.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_regressor_simple() {
        // Linear-ish data: left half low, right half high
        let X = array![
            [1.0], [2.0], [3.0], [4.0],
            [5.0], [6.0], [7.0], [8.0]
        ];
        let y = vec![1.0, 1.5, 2.0, 2.5, 10.0, 10.5, 11.0, 11.5];
        let mut tree = DecisionTreeRegressor::new(None, 2, 1, None, Some(42));
        tree.fit(&X.view(), &y);

        let preds = tree.predict(&X.view());
        // With enough depth, predictions should be close to targets
        for (pred, &actual) in preds.iter().zip(y.iter()) {
            assert!(
                (pred - actual).abs() < 2.0,
                "pred={}, actual={}",
                pred,
                actual
            );
        }
    }

    #[test]
    fn test_regressor_max_depth() {
        let X = array![
            [1.0], [2.0], [3.0], [4.0],
            [5.0], [6.0], [7.0], [8.0]
        ];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut tree = DecisionTreeRegressor::new(Some(1), 2, 1, None, Some(42));
        tree.fit(&X.view(), &y);

        let preds = tree.predict(&X.view());
        // With max_depth=1, we get exactly 2 distinct predictions (two leaves)
        let mut unique_preds = preds.clone();
        unique_preds.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        unique_preds.dedup();
        assert_eq!(unique_preds.len(), 2);
    }

    #[test]
    fn test_regressor_perfect_fit() {
        // With unlimited depth and 1 sample per leaf, tree should perfectly memorize
        let X = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let y = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let mut tree = DecisionTreeRegressor::new(None, 2, 1, None, Some(42));
        tree.fit(&X.view(), &y);

        let preds = tree.predict(&X.view());
        for (pred, &actual) in preds.iter().zip(y.iter()) {
            assert!(
                (pred - actual).abs() < 1e-6,
                "pred={}, actual={}",
                pred,
                actual
            );
        }
    }

    #[test]
    fn test_select_features() {
        let mut rng = SmallRng::seed_from_u64(42);
        let features = select_features(10, Some(5), &mut rng);
        assert_eq!(features.len(), 5);
        // All values should be in 0..10
        for &f in &features {
            assert!(f < 10);
        }
    }

    #[test]
    fn test_select_features_no_limit() {
        let mut rng = SmallRng::seed_from_u64(42);
        let features = select_features(10, None, &mut rng);
        assert_eq!(features.len(), 10);
    }

    #[test]
    fn test_tree_node_is_leaf() {
        let leaf = TreeNode::leaf(vec![1.0], 5);
        assert!(leaf.is_leaf());

        let internal = TreeNode {
            feature_idx: 0,
            threshold: 0.5,
            left: 1,
            right: 2,
            value: vec![0.5, 0.5],
            n_samples: 10,
        };
        assert!(!internal.is_leaf());
    }

    #[test]
    fn test_classifier_two_features() {
        // XOR-like pattern requires depth >= 2
        let X = array![
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0]
        ];
        let y = vec![0.0, 1.0, 1.0, 0.0];
        let mut tree = DecisionTreeClassifier::new(None, 2, 1, None, Some(42));
        tree.fit(&X.view(), &y);

        let preds = tree.predict(&X.view());
        assert_eq!(preds, y);
    }

    #[test]
    fn test_regressor_two_features() {
        let X = array![
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0]
        ];
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let mut tree = DecisionTreeRegressor::new(None, 2, 1, None, Some(42));
        tree.fit(&X.view(), &y);

        let preds = tree.predict(&X.view());
        for (pred, &actual) in preds.iter().zip(y.iter()) {
            assert!((pred - actual).abs() < 1e-6);
        }
    }
}
