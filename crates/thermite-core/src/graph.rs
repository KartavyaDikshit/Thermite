use ndarray::{Array1, Array2};
use rand::Rng;
use rand::seq::SliceRandom;

pub struct Node2Vec {
    pub p: f64,
    pub q: f64,
    pub walk_length: usize,
    pub num_walks: usize,
    pub embedding_dim: usize,
    pub embeddings: Option<Array2<f64>>,
}

impl Node2Vec {
    pub fn new(p: f64, q: f64, walk_length: usize, num_walks: usize, embedding_dim: usize) -> Self {
        Node2Vec {
            p,
            q,
            walk_length,
            num_walks,
            embedding_dim,
            embeddings: None,
        }
    }

    pub fn fit(&mut self, adjacency_list: &std::collections::HashMap<usize, Vec<usize>>) -> Result<(), String> {
        let mut rng = rand::thread_rng();
        let nodes: Vec<usize> = adjacency_list.keys().copied().collect();
        let num_nodes = nodes.len();
        if num_nodes == 0 {
            return Err("Empty graph".to_string());
        }

        let max_node_id = *nodes.iter().max().unwrap();
        let n_nodes = max_node_id + 1;

        // Generate random walks
        let mut walks = Vec::new();
        for _ in 0..self.num_walks {
            let mut shuffled_nodes = nodes.clone();
            shuffled_nodes.shuffle(&mut rng);
            for &node in &shuffled_nodes {
                let mut walk = vec![node];
                for _ in 1..self.walk_length {
                    let curr = *walk.last().unwrap();
                    if let Some(neighbors) = adjacency_list.get(&curr) {
                        if neighbors.is_empty() { break; }
                        let next = neighbors.choose(&mut rng).unwrap();
                        walk.push(*next);
                    } else { break; }
                }
                walks.push(walk);
            }
        }

        // Skip-gram with negative sampling
        let mut embeddings = Array2::<f64>::zeros((max_node_id + 1, self.embedding_dim));
        let mut context_embeddings = Array2::<f64>::zeros((max_node_id + 1, self.embedding_dim));
        let window_size = 5;
        let negative_samples = 5;

        for walk in &walks {
            for (pos, &node) in walk.iter().enumerate() {
                let start = if pos > window_size { pos - window_size } else { 0 };
                let end = (pos + window_size + 1).min(walk.len());
                for ctx_pos in start..end {
                    if ctx_pos == pos { continue; }
                    let ctx = walk[ctx_pos];

                    // Positive sample
                    let mut grad = Array1::<f64>::zeros(self.embedding_dim);
                    let dot = (0..self.embedding_dim).map(|f| embeddings[[node, f]] * context_embeddings[[ctx, f]]).sum::<f64>();
                    let sigmoid = 1.0 / (1.0 + (-dot).exp());
                    for f in 0..self.embedding_dim {
                        grad[f] = (sigmoid - 1.0) * context_embeddings[[ctx, f]];
                    }
                    for f in 0..self.embedding_dim {
                        embeddings[[node, f]] -= 0.01 * grad[f];
                        context_embeddings[[ctx, f]] -= 0.01 * (sigmoid - 1.0) * embeddings[[node, f]];
                    }

                    // Negative samples
                    for _ in 0..negative_samples {
                        let neg = rng.gen_range(0..n_nodes);
                        let dot = (0..self.embedding_dim).map(|f| embeddings[[node, f]] * context_embeddings[[neg, f]]).sum::<f64>();
                        let sigmoid = 1.0 / (1.0 + (-dot).exp());
                        for f in 0..self.embedding_dim {
                            embeddings[[node, f]] -= 0.01 * sigmoid * context_embeddings[[neg, f]];
                            context_embeddings[[neg, f]] -= 0.01 * sigmoid * embeddings[[node, f]];
                        }
                    }
                }
            }
        }

        self.embeddings = Some(embeddings);

        Ok(())
    }

    pub fn get_embeddings(&self) -> Result<Array2<f64>, String> {
        self.embeddings.clone().ok_or_else(|| "Model not fitted".to_string())
    }
}
