use ndarray::{Array1, Array2};

pub struct VectorStore {
    pub vectors: Array2<f64>,
}

impl VectorStore {
    pub fn new(vectors: Array2<f64>) -> Self {
        VectorStore { vectors }
    }

    pub fn search(&self, query: &Array1<f64>, k: usize) -> Result<(Vec<usize>, Vec<f64>), String> {
        let n = self.vectors.nrows();
        let m = self.vectors.ncols();
        if query.len() != m {
            return Err("Query dimension does not match vector store".to_string());
        }

        let mut distances = Vec::with_capacity(n);
        for i in 0..n {
            let row = self.vectors.row(i);
            let mut dist = 0.0;
            for j in 0..m {
                let diff = row[j] - query[j];
                dist += diff * diff;
            }
            distances.push((i, dist.sqrt()));
        }

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        
        let k = k.min(n);
        let mut top_indices = Vec::with_capacity(k);
        let mut top_distances = Vec::with_capacity(k);
        for i in 0..k {
            top_indices.push(distances[i].0);
            top_distances.push(distances[i].1);
        }

        Ok((top_indices, top_distances))
    }
}
