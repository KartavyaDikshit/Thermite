use std::collections::{HashMap, HashSet};
use ndarray::Array2;

#[derive(Clone)]
pub struct CountVectorizer {
    pub vocabulary: HashMap<String, usize>,
    pub lowercase: bool,
}

impl CountVectorizer {
    pub fn new(lowercase: bool) -> Self {
        Self {
            vocabulary: HashMap::new(),
            lowercase,
        }
    }

    pub fn fit(&mut self, docs: &[String]) {
        let mut idx = self.vocabulary.len();
        for doc in docs {
            let text = if self.lowercase { doc.to_lowercase() } else { doc.clone() };
            for token in text.split_whitespace() {
                if !self.vocabulary.contains_key(token) {
                    self.vocabulary.insert(token.to_string(), idx);
                    idx += 1;
                }
            }
        }
    }

    pub fn transform(&self, docs: &[String]) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((docs.len(), self.vocabulary.len()));
        for (i, doc) in docs.iter().enumerate() {
            let text = if self.lowercase { doc.to_lowercase() } else { doc.clone() };
            for token in text.split_whitespace() {
                if let Some(&j) = self.vocabulary.get(token) {
                    out[[i, j]] += 1.0;
                }
            }
        }
        out
    }

    pub fn fit_transform(&mut self, docs: &[String]) -> Array2<f64> {
        self.fit(docs);
        self.transform(docs)
    }
}

#[derive(Clone)]
pub struct TfidfVectorizer {
    pub count_vec: CountVectorizer,
    pub idf: Vec<f64>,
}

impl TfidfVectorizer {
    pub fn new(lowercase: bool) -> Self {
        Self {
            count_vec: CountVectorizer::new(lowercase),
            idf: Vec::new(),
        }
    }

    pub fn fit(&mut self, docs: &[String]) {
        self.count_vec.fit(docs);
        let n_docs = docs.len() as f64;
        let vocab_size = self.count_vec.vocabulary.len();
        let mut df = vec![0.0; vocab_size];

        for doc in docs {
            let text = if self.count_vec.lowercase { doc.to_lowercase() } else { doc.clone() };
            let mut seen = HashSet::new();
            for token in text.split_whitespace() {
                if let Some(&j) = self.count_vec.vocabulary.get(token) {
                    if seen.insert(j) {
                        df[j] += 1.0;
                    }
                }
            }
        }

        // smoothing: ln((1 + n_docs) / (1 + df)) + 1
        self.idf = df.into_iter().map(|d| ((1.0 + n_docs) / (1.0 + d)).ln() + 1.0).collect();
    }

    pub fn transform(&self, docs: &[String]) -> Array2<f64> {
        let mut tf = self.count_vec.transform(docs);
        for mut row in tf.rows_mut() {
            for j in 0..row.len() {
                row[j] *= self.idf[j];
            }
            // L2 normalize
            let norm = row.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 0.0 {
                for j in 0..row.len() {
                    row[j] /= norm;
                }
            }
        }
        tf
    }

    pub fn fit_transform(&mut self, docs: &[String]) -> Array2<f64> {
        self.fit(docs);
        self.transform(docs)
    }
}

pub struct Word2Vec {
    pub vector_size: usize,
    pub window: usize,
    pub min_count: usize,
    pub embeddings: Option<HashMap<String, Vec<f64>>>,
}

impl Word2Vec {
    pub fn new(vector_size: usize, window: usize, min_count: usize) -> Self {
        Word2Vec {
            vector_size,
            window,
            min_count,
            embeddings: None,
        }
    }

    pub fn fit(&mut self, sentences: &[Vec<String>]) -> Result<(), String> {
        let mut counts: HashMap<String, usize> = HashMap::new();
        for sentence in sentences {
            for word in sentence {
                *counts.entry(word.clone()).or_insert(0) += 1;
            }
        }

        let vocab: Vec<String> = counts.into_iter()
            .filter(|(_, c)| *c >= self.min_count)
            .map(|(w, _)| w)
            .collect();

        if vocab.is_empty() {
            return Err("Empty vocabulary".to_string());
        }

        // Simplified placeholder: random embeddings
        let mut embeddings = HashMap::new();
        for word in vocab {
            let mut vec = vec![0.0; self.vector_size];
            for v in &mut vec {
                *v = (rand::random::<f64>() * 2.0) - 1.0;
            }
            embeddings.insert(word, vec);
        }
        self.embeddings = Some(embeddings);

        Ok(())
    }

    pub fn get_embeddings(&self) -> Result<HashMap<String, Vec<f64>>, String> {
        self.embeddings.clone().ok_or_else(|| "Model not fitted".to_string())
    }
}
