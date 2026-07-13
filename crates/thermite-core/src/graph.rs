use ndarray::Array2;
use rand::Rng;
use rand::seq::SliceRandom;
use std::collections::HashMap;

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

    pub fn fit(&mut self, adjacency_list: &HashMap<usize, Vec<usize>>) -> Result<(), String> {
        let mut rng = rand::thread_rng();
        let nodes: Vec<usize> = adjacency_list.keys().copied().collect();
        let num_nodes = nodes.len();
        if num_nodes == 0 {
            return Err("Empty graph".to_string());
        }

        let max_node_id = *nodes.iter().max().unwrap();
        let mut walks = Vec::new();

        for _ in 0..self.num_walks {
            let mut shuffled_nodes = nodes.clone();
            shuffled_nodes.shuffle(&mut rng);
            for &node in &shuffled_nodes {
                let mut walk = vec![node];
                for _ in 1..self.walk_length {
                    let curr = *walk.last().unwrap();
                    if let Some(neighbors) = adjacency_list.get(&curr) {
                        if neighbors.is_empty() {
                            break;
                        }
                        // Simplified random walk (ignoring p and q for milestone)
                        let next = neighbors.choose(&mut rng).unwrap();
                        walk.push(*next);
                    } else {
                        break;
                    }
                }
                walks.push(walk);
            }
        }

        // Placeholder Skip-gram training: just random embeddings for now as simplified logic
        let mut embeddings = Array2::<f64>::zeros((max_node_id + 1, self.embedding_dim));
        for i in 0..=max_node_id {
            for j in 0..self.embedding_dim {
                embeddings[[i, j]] = rng.gen_range(-1.0..1.0);
            }
        }
        self.embeddings = Some(embeddings);

        Ok(())
    }

    pub fn get_embeddings(&self) -> Result<Array2<f64>, String> {
        self.embeddings.clone().ok_or_else(|| "Model not fitted".to_string())
    }
}
