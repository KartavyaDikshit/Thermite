#![allow(non_snake_case)]

use std::collections::HashMap;

// ==========================================
// Helper: collect unique classes
// ==========================================
fn unique_classes(y: &[f64]) -> Vec<f64> {
    let mut classes: Vec<f64> = Vec::new();
    for &v in y {
        if !classes.iter().any(|&c| (c - v).abs() < f64::EPSILON) {
            classes.push(v);
        }
    }
    classes.sort_unstable_by(|a, b| a.total_cmp(b));
    classes
}

/// Hash-friendly wrapper for f64 class labels (assumes labels are integer-like).
fn class_key(v: f64) -> i64 {
    v.round() as i64
}

// ==========================================
// Accuracy
// ==========================================
/// Fraction of correctly classified samples.
pub fn accuracy_score(y_true: &[f64], y_pred: &[f64]) -> Result<f64, String> {
    if y_true.len() != y_pred.len() {
        return Err(format!(
            "Length mismatch: y_true has {} elements, y_pred has {}",
            y_true.len(),
            y_pred.len()
        ));
    }
    if y_true.is_empty() {
        return Err("Input arrays are empty".to_string());
    }
    let correct = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|(&t, &p)| (t - p).abs() < f64::EPSILON)
        .count();
    Ok(correct as f64 / y_true.len() as f64)
}

// ==========================================
// Per-class precision / recall helpers
// ==========================================
struct PerClassStats {
    tp: HashMap<i64, usize>,
    fp: HashMap<i64, usize>,
    fn_: HashMap<i64, usize>,
    support: HashMap<i64, usize>,
}

fn compute_per_class(y_true: &[f64], y_pred: &[f64]) -> Result<(Vec<f64>, PerClassStats), String> {
    if y_true.len() != y_pred.len() {
        return Err(format!(
            "Length mismatch: y_true has {} elements, y_pred has {}",
            y_true.len(),
            y_pred.len()
        ));
    }
    if y_true.is_empty() {
        return Err("Input arrays are empty".to_string());
    }

    let classes = unique_classes(y_true);
    let mut tp: HashMap<i64, usize> = HashMap::new();
    let mut fp: HashMap<i64, usize> = HashMap::new();
    let mut fn_: HashMap<i64, usize> = HashMap::new();
    let mut support: HashMap<i64, usize> = HashMap::new();

    for &c in &classes {
        let k = class_key(c);
        tp.insert(k, 0);
        fp.insert(k, 0);
        fn_.insert(k, 0);
        support.insert(k, 0);
    }

    for (&t, &p) in y_true.iter().zip(y_pred.iter()) {
        let tk = class_key(t);
        let pk = class_key(p);
        *support.entry(tk).or_insert(0) += 1;
        if tk == pk {
            *tp.entry(tk).or_insert(0) += 1;
        } else {
            *fn_.entry(tk).or_insert(0) += 1;
            *fp.entry(pk).or_insert(0) += 1;
        }
    }

    Ok((
        classes,
        PerClassStats {
            tp,
            fp,
            fn_,
            support,
        },
    ))
}

// ==========================================
// Precision
// ==========================================
/// Precision score with averaging: "binary", "macro", "weighted".
pub fn precision_score(y_true: &[f64], y_pred: &[f64], average: &str) -> Result<f64, String> {
    let (classes, stats) = compute_per_class(y_true, y_pred)?;

    match average {
        "binary" => {
            if classes.len() > 2 {
                return Err("binary average requires at most 2 classes".to_string());
            }
            // Positive class is the largest label
            let pos = class_key(*classes.last().unwrap());
            let tp = *stats.tp.get(&pos).unwrap_or(&0);
            let fp = *stats.fp.get(&pos).unwrap_or(&0);
            if tp + fp == 0 {
                Ok(0.0)
            } else {
                Ok(tp as f64 / (tp + fp) as f64)
            }
        }
        "macro" => {
            let n = classes.len() as f64;
            let sum: f64 = classes
                .iter()
                .map(|&c| {
                    let k = class_key(c);
                    let tp = *stats.tp.get(&k).unwrap_or(&0);
                    let fp = *stats.fp.get(&k).unwrap_or(&0);
                    if tp + fp == 0 {
                        0.0
                    } else {
                        tp as f64 / (tp + fp) as f64
                    }
                })
                .sum();
            Ok(sum / n)
        }
        "weighted" => {
            let total: usize = stats.support.values().sum();
            if total == 0 {
                return Ok(0.0);
            }
            let sum: f64 = classes
                .iter()
                .map(|&c| {
                    let k = class_key(c);
                    let tp = *stats.tp.get(&k).unwrap_or(&0);
                    let fp = *stats.fp.get(&k).unwrap_or(&0);
                    let sup = *stats.support.get(&k).unwrap_or(&0);
                    let prec = if tp + fp == 0 {
                        0.0
                    } else {
                        tp as f64 / (tp + fp) as f64
                    };
                    prec * sup as f64
                })
                .sum();
            Ok(sum / total as f64)
        }
        _ => Err(format!(
            "Unsupported average: '{}'. Use 'binary', 'macro', or 'weighted'.",
            average
        )),
    }
}

// ==========================================
// Recall
// ==========================================
/// Recall score with averaging: "binary", "macro", "weighted".
pub fn recall_score(y_true: &[f64], y_pred: &[f64], average: &str) -> Result<f64, String> {
    let (classes, stats) = compute_per_class(y_true, y_pred)?;

    match average {
        "binary" => {
            if classes.len() > 2 {
                return Err("binary average requires at most 2 classes".to_string());
            }
            let pos = class_key(*classes.last().unwrap());
            let tp = *stats.tp.get(&pos).unwrap_or(&0);
            let fn_ = *stats.fn_.get(&pos).unwrap_or(&0);
            if tp + fn_ == 0 {
                Ok(0.0)
            } else {
                Ok(tp as f64 / (tp + fn_) as f64)
            }
        }
        "macro" => {
            let n = classes.len() as f64;
            let sum: f64 = classes
                .iter()
                .map(|&c| {
                    let k = class_key(c);
                    let tp = *stats.tp.get(&k).unwrap_or(&0);
                    let fn_ = *stats.fn_.get(&k).unwrap_or(&0);
                    if tp + fn_ == 0 {
                        0.0
                    } else {
                        tp as f64 / (tp + fn_) as f64
                    }
                })
                .sum();
            Ok(sum / n)
        }
        "weighted" => {
            let total: usize = stats.support.values().sum();
            if total == 0 {
                return Ok(0.0);
            }
            let sum: f64 = classes
                .iter()
                .map(|&c| {
                    let k = class_key(c);
                    let tp = *stats.tp.get(&k).unwrap_or(&0);
                    let fn_ = *stats.fn_.get(&k).unwrap_or(&0);
                    let sup = *stats.support.get(&k).unwrap_or(&0);
                    let rec = if tp + fn_ == 0 {
                        0.0
                    } else {
                        tp as f64 / (tp + fn_) as f64
                    };
                    rec * sup as f64
                })
                .sum();
            Ok(sum / total as f64)
        }
        _ => Err(format!(
            "Unsupported average: '{}'. Use 'binary', 'macro', or 'weighted'.",
            average
        )),
    }
}

// ==========================================
// F1 Score
// ==========================================
/// F1 score with averaging: "binary", "macro", "weighted".
pub fn f1_score(y_true: &[f64], y_pred: &[f64], average: &str) -> Result<f64, String> {
    let (classes, stats) = compute_per_class(y_true, y_pred)?;

    let per_class_f1 = |k: i64| -> f64 {
        let tp = *stats.tp.get(&k).unwrap_or(&0) as f64;
        let fp = *stats.fp.get(&k).unwrap_or(&0) as f64;
        let fn_ = *stats.fn_.get(&k).unwrap_or(&0) as f64;
        let prec = if tp + fp == 0.0 { 0.0 } else { tp / (tp + fp) };
        let rec = if tp + fn_ == 0.0 {
            0.0
        } else {
            tp / (tp + fn_)
        };
        if prec + rec == 0.0 {
            0.0
        } else {
            2.0 * prec * rec / (prec + rec)
        }
    };

    match average {
        "binary" => {
            if classes.len() > 2 {
                return Err("binary average requires at most 2 classes".to_string());
            }
            let pos = class_key(*classes.last().unwrap());
            Ok(per_class_f1(pos))
        }
        "macro" => {
            let n = classes.len() as f64;
            let sum: f64 = classes.iter().map(|&c| per_class_f1(class_key(c))).sum();
            Ok(sum / n)
        }
        "weighted" => {
            let total: usize = stats.support.values().sum();
            if total == 0 {
                return Ok(0.0);
            }
            let sum: f64 = classes
                .iter()
                .map(|&c| {
                    let k = class_key(c);
                    let sup = *stats.support.get(&k).unwrap_or(&0);
                    per_class_f1(k) * sup as f64
                })
                .sum();
            Ok(sum / total as f64)
        }
        _ => Err(format!(
            "Unsupported average: '{}'. Use 'binary', 'macro', or 'weighted'.",
            average
        )),
    }
}

// ==========================================
// ROC AUC Score (binary only, using trapezoidal rule)
// ==========================================
/// ROC AUC for binary classification.
/// `y_true` must contain only 0.0 and 1.0 labels.
/// `y_score` contains the predicted probability / score for the positive class.
pub fn roc_auc_score(y_true: &[f64], y_score: &[f64]) -> Result<f64, String> {
    if y_true.len() != y_score.len() {
        return Err(format!(
            "Length mismatch: y_true has {} elements, y_score has {}",
            y_true.len(),
            y_score.len()
        ));
    }
    if y_true.is_empty() {
        return Err("Input arrays are empty".to_string());
    }

    let n_pos = y_true
        .iter()
        .filter(|&&v| (v - 1.0).abs() < f64::EPSILON)
        .count();
    let n_neg = y_true.len() - n_pos;
    if n_pos == 0 || n_neg == 0 {
        return Err("ROC AUC requires both positive and negative samples".to_string());
    }

    // Sort by descending score
    let mut indices: Vec<usize> = (0..y_true.len()).collect();
    indices.sort_unstable_by(|&a, &b| y_score[b].total_cmp(&y_score[a]));

    // Compute AUC via trapezoidal rule on the ROC curve
    let mut tp = 0.0_f64;
    let mut fp = 0.0_f64;
    let mut auc = 0.0_f64;
    let mut prev_tp = 0.0_f64;
    let mut prev_fp = 0.0_f64;
    let mut prev_score = f64::NAN;

    for &idx in &indices {
        let score = y_score[idx];
        // If score changes, add the trapezoid for the previous block
        if !prev_score.is_nan() && (score - prev_score).abs() > f64::EPSILON {
            auc += (fp - prev_fp) * (tp + prev_tp) / 2.0;
            prev_tp = tp;
            prev_fp = fp;
        }
        if (y_true[idx] - 1.0).abs() < f64::EPSILON {
            tp += 1.0;
        } else {
            fp += 1.0;
        }
        prev_score = score;
    }
    // Final trapezoid
    auc += (fp - prev_fp) * (tp + prev_tp) / 2.0;

    Ok(auc / (n_pos as f64 * n_neg as f64))
}

// ==========================================
// Mean Squared Error
// ==========================================
pub fn mean_squared_error(y_true: &[f64], y_pred: &[f64]) -> Result<f64, String> {
    if y_true.len() != y_pred.len() {
        return Err(format!(
            "Length mismatch: y_true has {} elements, y_pred has {}",
            y_true.len(),
            y_pred.len()
        ));
    }
    if y_true.is_empty() {
        return Err("Input arrays are empty".to_string());
    }
    let mse: f64 = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&t, &p)| (t - p).powi(2))
        .sum::<f64>()
        / y_true.len() as f64;
    Ok(mse)
}

// ==========================================
// R² Score
// ==========================================
pub fn r2_score(y_true: &[f64], y_pred: &[f64]) -> Result<f64, String> {
    if y_true.len() != y_pred.len() {
        return Err(format!(
            "Length mismatch: y_true has {} elements, y_pred has {}",
            y_true.len(),
            y_pred.len()
        ));
    }
    if y_true.len() < 2 {
        return Err("R² score requires at least 2 samples".to_string());
    }
    let mean_y = y_true.iter().sum::<f64>() / y_true.len() as f64;
    let ss_res: f64 = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&t, &p)| (t - p).powi(2))
        .sum();
    let ss_tot: f64 = y_true.iter().map(|&t| (t - mean_y).powi(2)).sum();
    if ss_tot == 0.0 {
        // All true values are identical; R² is undefined, return 0 by convention
        return Ok(0.0);
    }
    Ok(1.0 - ss_res / ss_tot)
}

// ==========================================
// Log Loss
// ==========================================
pub fn log_loss(y_true: &[f64], y_pred: &[f64]) -> Result<f64, String> {
    if y_true.len() != y_pred.len() {
        return Err(format!(
            "Length mismatch: y_true has {} elements, y_pred has {}",
            y_true.len(),
            y_pred.len()
        ));
    }
    if y_true.is_empty() {
        return Err("Input arrays are empty".to_string());
    }
    let eps = 1e-15;
    let loss: f64 = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&t, &p)| {
            let p_clip = p.clamp(eps, 1.0 - eps);
            if (t - 1.0).abs() < f64::EPSILON {
                -p_clip.ln()
            } else if t.abs() < f64::EPSILON {
                -(1.0 - p_clip).ln()
            } else {
                -(t * p_clip.ln() + (1.0 - t) * (1.0 - p_clip).ln())
            }
        })
        .sum();
    Ok(loss / y_true.len() as f64)
}

// ==========================================
// Mean Absolute Percentage Error
// ==========================================
pub fn mean_absolute_percentage_error(y_true: &[f64], y_pred: &[f64]) -> Result<f64, String> {
    if y_true.len() != y_pred.len() {
        return Err(format!(
            "Length mismatch: y_true has {} elements, y_pred has {}",
            y_true.len(),
            y_pred.len()
        ));
    }
    if y_true.is_empty() {
        return Err("Input arrays are empty".to_string());
    }
    let eps = f64::EPSILON;
    let mape: f64 = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&t, &p)| {
            if t.abs() < eps {
                ((t - p).abs() / eps.max(t.abs()))
            } else {
                ((t - p) / t).abs()
            }
        })
        .sum();
    Ok(mape / y_true.len() as f64)
}

// ==========================================
// Pairwise Distances
// ==========================================
pub fn pairwise_distances(
    x: &[f64],
    n_samples_x: usize,
    y: &[f64],
    n_samples_y: usize,
    n_features: usize,
    metric: &str,
) -> Result<Vec<f64>, String> {
    if x.len() != n_samples_x * n_features || y.len() != n_samples_y * n_features {
        return Err("Dimensions do not match".to_string());
    }
    let mut distances = Vec::with_capacity(n_samples_x * n_samples_y);
    for i in 0..n_samples_x {
        let x_row = &x[i * n_features..(i + 1) * n_features];
        for j in 0..n_samples_y {
            let y_row = &y[j * n_features..(j + 1) * n_features];
            match metric {
                "cosine" => {
                    let mut dot = 0.0;
                    let mut norm_x = 0.0;
                    let mut norm_y = 0.0;
                    for (&xi, &yi) in x_row.iter().zip(y_row.iter()) {
                        dot += xi * yi;
                        norm_x += xi * xi;
                        norm_y += yi * yi;
                    }
                    if norm_x == 0.0 || norm_y == 0.0 {
                        distances.push(1.0);
                    } else {
                        distances.push(1.0 - (dot / (norm_x.sqrt() * norm_y.sqrt())));
                    }
                }
                "manhattan" => {
                    let dist: f64 = x_row.iter().zip(y_row.iter()).map(|(xi, yi)| (xi - yi).abs()).sum();
                    distances.push(dist);
                }
                "haversine" => {
                    if n_features != 2 {
                        return Err("Haversine requires exactly 2 dimensions".to_string());
                    }
                    let lat1 = x_row[0];
                    let lon1 = x_row[1];
                    let lat2 = y_row[0];
                    let lon2 = y_row[1];
                    let dlat = lat2 - lat1;
                    let dlon = lon2 - lon1;
                    let a = (dlat / 2.0).sin().powi(2) + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().powi(2);
                    let c = 2.0 * a.sqrt().asin();
                    distances.push(c);
                }
                _ => return Err(format!("Unsupported metric: {}", metric))
            }
        }
    }
    Ok(distances)
}

// ==========================================
// Tests
// ==========================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accuracy() {
        let y_true = vec![0.0, 1.0, 1.0, 0.0, 1.0];
        let y_pred = vec![0.0, 1.0, 0.0, 0.0, 1.0];
        let acc = accuracy_score(&y_true, &y_pred).unwrap();
        assert!((acc - 0.8).abs() < 1e-9);
    }

    #[test]
    fn test_precision_binary() {
        // tp=2, fp=1 for positive class (1.0)
        let y_true = vec![0.0, 1.0, 1.0, 0.0, 1.0];
        let y_pred = vec![0.0, 1.0, 0.0, 1.0, 1.0];
        let prec = precision_score(&y_true, &y_pred, "binary").unwrap();
        // tp=2, fp=1 => 2/3
        assert!((prec - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_recall_binary() {
        let y_true = vec![0.0, 1.0, 1.0, 0.0, 1.0];
        let y_pred = vec![0.0, 1.0, 0.0, 1.0, 1.0];
        let rec = recall_score(&y_true, &y_pred, "binary").unwrap();
        // tp=2, fn=1 => 2/3
        assert!((rec - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_f1_binary() {
        let y_true = vec![0.0, 1.0, 1.0, 0.0, 1.0];
        let y_pred = vec![0.0, 1.0, 0.0, 1.0, 1.0];
        let f1 = f1_score(&y_true, &y_pred, "binary").unwrap();
        // prec = 2/3, rec = 2/3, f1 = 2/3
        assert!((f1 - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_f1_macro() {
        let y_true = vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0];
        let y_pred = vec![0.0, 2.0, 1.0, 0.0, 1.0, 2.0];
        let f1 = f1_score(&y_true, &y_pred, "macro").unwrap();
        // class 0: tp=2, fp=0, fn=0 => f1=1.0
        // class 1: tp=1, fp=1, fn=1 => prec=0.5, rec=0.5, f1=0.5
        // class 2: tp=1, fp=1, fn=1 => prec=0.5, rec=0.5, f1=0.5
        // macro = (1.0+0.5+0.5)/3 = 2/3
        assert!((f1 - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_roc_auc() {
        let y_true = vec![0.0, 0.0, 1.0, 1.0];
        let y_score = vec![0.1, 0.4, 0.35, 0.8];
        let auc = roc_auc_score(&y_true, &y_score).unwrap();
        // Perfect separation check: not perfect here
        assert!(auc > 0.5);
        assert!(auc < 1.0);
    }

    #[test]
    fn test_roc_auc_perfect() {
        let y_true = vec![0.0, 0.0, 1.0, 1.0];
        let y_score = vec![0.1, 0.2, 0.8, 0.9];
        let auc = roc_auc_score(&y_true, &y_score).unwrap();
        assert!((auc - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_mse() {
        let y_true = vec![3.0, -0.5, 2.0, 7.0];
        let y_pred = vec![2.5, 0.0, 2.0, 8.0];
        let mse = mean_squared_error(&y_true, &y_pred).unwrap();
        // (0.25 + 0.25 + 0 + 1) / 4 = 0.375
        assert!((mse - 0.375).abs() < 1e-9);
    }

    #[test]
    fn test_r2() {
        let y_true = vec![3.0, -0.5, 2.0, 7.0];
        let y_pred = vec![2.5, 0.0, 2.0, 8.0];
        let r2 = r2_score(&y_true, &y_pred).unwrap();
        // sklearn gives 0.9486081370449679
        assert!((r2 - 0.9486081370449679).abs() < 1e-7);
    }

    #[test]
    fn test_length_mismatch() {
        assert!(accuracy_score(&[1.0, 2.0], &[1.0]).is_err());
        assert!(mean_squared_error(&[1.0], &[1.0, 2.0]).is_err());
        assert!(r2_score(&[1.0], &[1.0]).is_err()); // needs >= 2
    }

    #[test]
    fn test_empty_input() {
        assert!(accuracy_score(&[], &[]).is_err());
        assert!(mean_squared_error(&[], &[]).is_err());
    }
}
