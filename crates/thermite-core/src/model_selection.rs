use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct SplitIndices {
    pub train_indices: Vec<usize>,
    pub test_indices: Vec<usize>,
}

struct ClassSplit<'a> {
    n_c: usize,
    test_c: usize,
    train_c: usize,
    test_fract: f64,
    train_fract: f64,
    group_idx: &'a Vec<usize>,
}

pub fn compute_split_indices(
    n_samples: usize,
    test_size: Option<f64>,
    train_size: Option<f64>,
    shuffle: bool,
    random_state: Option<u64>,
    stratify: Option<&[String]>,
) -> Result<SplitIndices, String> {
    if n_samples == 0 {
        return Err("Number of samples must be greater than 0".to_string());
    }

    let (n_train, n_test) = determine_split_sizes(n_samples, test_size, train_size)?;

    let mut rng = match random_state {
        Some(seed) => SmallRng::seed_from_u64(seed),
        None => SmallRng::from_entropy(),
    };

    if let Some(stratify_labels) = stratify {
        if stratify_labels.len() != n_samples {
            return Err("Stratify labels length must match number of samples".to_string());
        }

        // Group indices by label
        let mut label_to_indices: HashMap<String, Vec<usize>> = HashMap::new();
        for (idx, label) in stratify_labels.iter().enumerate() {
            label_to_indices.entry(label.clone()).or_default().push(idx);
        }

        // Shuffle each group if shuffle is enabled
        if shuffle {
            for indices_group in label_to_indices.values_mut() {
                indices_group.shuffle(&mut rng);
            }
        }

        let test_ratio = n_test as f64 / n_samples as f64;
        let train_ratio = n_train as f64 / n_samples as f64;

        let mut class_splits = Vec::new();
        let mut allocated_test = 0;
        let mut allocated_train = 0;

        for (_label, group_idx) in label_to_indices.iter() {
            let n_c = group_idx.len();
            let test_c_exact = n_c as f64 * test_ratio;
            let test_c = test_c_exact.floor() as usize;
            let test_fract = test_c_exact - test_c as f64;

            let train_c_exact = n_c as f64 * train_ratio;
            let train_c = train_c_exact.floor() as usize;
            let train_fract = train_c_exact - train_c as f64;

            class_splits.push(ClassSplit {
                n_c,
                test_c,
                train_c,
                test_fract,
                train_fract,
                group_idx,
            });

            allocated_test += test_c;
            allocated_train += train_c;
        }

        // Distribute remaining test slots using Largest Remainder Method
        if allocated_test < n_test {
            class_splits.sort_by(|a, b| b.test_fract.partial_cmp(&a.test_fract).unwrap());
            let mut diff = n_test - allocated_test;
            for split in class_splits.iter_mut() {
                if diff == 0 {
                    break;
                }
                let unallocated = split.n_c - (split.test_c + split.train_c);
                if unallocated > 0 {
                    split.test_c += 1;
                    diff -= 1;
                }
            }
        }

        // Distribute remaining train slots using Largest Remainder Method
        if allocated_train < n_train {
            class_splits.sort_by(|a, b| b.train_fract.partial_cmp(&a.train_fract).unwrap());
            let mut diff = n_train - allocated_train;
            for split in class_splits.iter_mut() {
                if diff == 0 {
                    break;
                }
                let unallocated = split.n_c - (split.test_c + split.train_c);
                if unallocated > 0 {
                    split.train_c += 1;
                    diff -= 1;
                }
            }
        }

        let mut train_indices = Vec::with_capacity(n_train);
        let mut test_indices = Vec::with_capacity(n_test);

        for split in class_splits {
            let group_idx = split.group_idx;
            let test_c = split.test_c;
            let train_c = split.train_c;

            test_indices.extend_from_slice(&group_idx[0..test_c]);
            train_indices.extend_from_slice(&group_idx[test_c..test_c + train_c]);
        }

        if shuffle {
            train_indices.shuffle(&mut rng);
            test_indices.shuffle(&mut rng);
        }

        Ok(SplitIndices {
            train_indices,
            test_indices,
        })
    } else {
        let mut indices: Vec<usize> = (0..n_samples).collect();
        if shuffle {
            indices.shuffle(&mut rng);
        }
        let (test_part, train_part) = indices.split_at(n_test);
        let mut train_indices = train_part.to_vec();
        // Since indices.split_at(n_test) returns elements from n_test to n_samples,
        // if n_train + n_test < n_samples, train_indices will have length n_samples - n_test.
        // We must truncate it to n_train.
        train_indices.truncate(n_train);

        Ok(SplitIndices {
            train_indices,
            test_indices: test_part.to_vec(),
        })
    }
}

fn determine_split_sizes(
    n_samples: usize,
    test_size: Option<f64>,
    train_size: Option<f64>,
) -> Result<(usize, usize), String> {
    let default_test_size = 0.25;

    let (n_train, n_test) = match (test_size, train_size) {
        (None, None) => {
            let n_test = (n_samples as f64 * default_test_size).round() as usize;
            let n_train = n_samples - n_test;
            (n_train, n_test)
        }
        (Some(ts), None) => {
            let n_test = if ts >= 1.0 {
                ts as usize
            } else if ts > 0.0 && ts < 1.0 {
                (n_samples as f64 * ts).round() as usize
            } else {
                return Err("test_size must be > 0".to_string());
            };
            if n_test > n_samples {
                return Err(format!(
                    "test_size={} is greater than n_samples={}",
                    n_test, n_samples
                ));
            }
            (n_samples - n_test, n_test)
        }
        (None, Some(tr)) => {
            let n_train = if tr >= 1.0 {
                tr as usize
            } else if tr > 0.0 && tr < 1.0 {
                (n_samples as f64 * tr).round() as usize
            } else {
                return Err("train_size must be > 0".to_string());
            };
            if n_train > n_samples {
                return Err(format!(
                    "train_size={} is greater than n_samples={}",
                    n_train, n_samples
                ));
            }
            (n_train, n_samples - n_train)
        }
        (Some(ts), Some(tr)) => {
            let n_test = if ts >= 1.0 {
                ts as usize
            } else if ts > 0.0 && ts < 1.0 {
                (n_samples as f64 * ts).round() as usize
            } else {
                return Err("test_size must be > 0".to_string());
            };
            let n_train = if tr >= 1.0 {
                tr as usize
            } else if tr > 0.0 && tr < 1.0 {
                (n_samples as f64 * tr).round() as usize
            } else {
                return Err("train_size must be > 0".to_string());
            };
            if n_test + n_train > n_samples {
                return Err(format!(
                    "The sum of train_size={} and test_size={} is larger than n_samples={}",
                    n_train, n_test, n_samples
                ));
            }
            (n_train, n_test)
        }
    };

    if n_train == 0 || n_test == 0 {
        return Err("Train or test size cannot be zero".to_string());
    }

    Ok((n_train, n_test))
}

pub struct StratifiedKFold {
    pub n_splits: usize,
    pub shuffle: bool,
    pub random_state: Option<u64>,
}

impl StratifiedKFold {
    pub fn split(&self, y: &[i64]) -> Result<Vec<SplitIndices>, String> {
        if self.n_splits < 2 {
            return Err("n_splits must be at least 2".to_string());
        }

        let mut class_indices: std::collections::HashMap<i64, Vec<usize>> = std::collections::HashMap::new();
        for (i, &label) in y.iter().enumerate() {
            class_indices.entry(label).or_default().push(i);
        }

        let mut rng = match self.random_state {
            Some(seed) => rand::rngs::SmallRng::seed_from_u64(seed),
            None => rand::rngs::SmallRng::from_entropy(),
        };

        if self.shuffle {
            for indices in class_indices.values_mut() {
                indices.shuffle(&mut rng);
            }
        }

        let mut folds = vec![Vec::new(); self.n_splits];
        
        for indices in class_indices.values() {
            let mut current_fold = 0;
            for &idx in indices {
                folds[current_fold].push(idx);
                current_fold = (current_fold + 1) % self.n_splits;
            }
        }

        let mut splits = Vec::with_capacity(self.n_splits);
        for i in 0..self.n_splits {
            let mut test_indices = Vec::new();
            let mut train_indices = Vec::new();

            for j in 0..self.n_splits {
                if i == j {
                    test_indices.extend_from_slice(&folds[j]);
                } else {
                    train_indices.extend_from_slice(&folds[j]);
                }
            }
            splits.push(SplitIndices { train_indices, test_indices });
        }

        Ok(splits)
    }
}

pub struct TimeSeriesSplit {
    pub n_splits: usize,
}

impl TimeSeriesSplit {
    pub fn split(&self, n_samples: usize) -> Result<Vec<SplitIndices>, String> {
        if self.n_splits < 2 {
            return Err("n_splits must be at least 2".to_string());
        }
        if n_samples < self.n_splits + 1 {
            return Err("n_samples must be greater than n_splits".to_string());
        }

        let fold_size = n_samples / (self.n_splits + 1);
        let remainder = n_samples % (self.n_splits + 1);
        
        let mut splits = Vec::with_capacity(self.n_splits);
        let mut offset = remainder;

        for i in 1..=self.n_splits {
            let train_end = offset + i * fold_size;
            let test_end = train_end + fold_size;

            let train_indices = (0..train_end).collect();
            let test_indices = (train_end..test_end).collect();

            splits.push(SplitIndices { train_indices, test_indices });
        }
        
        Ok(splits)
    }
}

pub struct GroupKFold {
    pub n_splits: usize,
}

impl GroupKFold {
    pub fn split(&self, groups: &[i64]) -> Result<Vec<SplitIndices>, String> {
        if self.n_splits < 2 {
            return Err("n_splits must be at least 2".to_string());
        }

        let mut group_to_indices: std::collections::HashMap<i64, Vec<usize>> = std::collections::HashMap::new();
        for (i, &g) in groups.iter().enumerate() {
            group_to_indices.entry(g).or_default().push(i);
        }
        
        let unique_groups: Vec<i64> = group_to_indices.keys().copied().collect();
        if unique_groups.len() < self.n_splits {
            return Err("n_splits cannot be greater than the number of groups".to_string());
        }

        let mut group_folds = vec![Vec::new(); self.n_splits];
        for (i, g) in unique_groups.iter().enumerate() {
            group_folds[i % self.n_splits].push(*g);
        }

        let mut splits = Vec::with_capacity(self.n_splits);
        for i in 0..self.n_splits {
            let mut test_indices = Vec::new();
            let mut train_indices = Vec::new();

            for j in 0..self.n_splits {
                for g in &group_folds[j] {
                    if i == j {
                        test_indices.extend_from_slice(&group_to_indices[g]);
                    } else {
                        train_indices.extend_from_slice(&group_to_indices[g]);
                    }
                }
            }
            splits.push(SplitIndices { train_indices, test_indices });
        }

        Ok(splits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_indices_basic() {
        let indices = compute_split_indices(100, Some(0.20), None, true, Some(42), None).unwrap();
        assert_eq!(indices.train_indices.len(), 80);
        assert_eq!(indices.test_indices.len(), 20);

        // Ensure no overlap
        for idx in &indices.train_indices {
            assert!(!indices.test_indices.contains(idx));
        }
    }

    #[test]
    fn test_split_indices_stratified() {
        let labels: Vec<String> = (0..100)
            .map(|i| if i % 2 == 0 { "A" } else { "B" }.to_string())
            .collect();
        let indices =
            compute_split_indices(100, Some(0.50), None, true, Some(42), Some(&labels)).unwrap();

        assert_eq!(indices.train_indices.len(), 50);
        assert_eq!(indices.test_indices.len(), 50);

        let train_a = indices
            .train_indices
            .iter()
            .filter(|&&i| labels[i] == "A")
            .count();
        let test_a = indices
            .test_indices
            .iter()
            .filter(|&&i| labels[i] == "A")
            .count();
        assert_eq!(train_a, 25);
        assert_eq!(test_a, 25);
    }
}
