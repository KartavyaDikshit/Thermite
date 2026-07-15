use ndarray::{Array1, Array2, ArrayView2};

fn pairwise_distances(X: &ArrayView2<f64>) -> Array2<f64> {
    let n = X.nrows();
    let mut D = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in (i + 1)..n {
            let mut d = 0.0;
            for k in 0..X.ncols() {
                let diff = X[[i, k]] - X[[j, k]];
                d += diff * diff;
            }
            d = d.sqrt();
            D[[i, j]] = d;
            D[[j, i]] = d;
        }
    }
    D
}

fn power_eigen(A: &Array2<f64>, n_components: usize) -> (Array2<f64>, Array1<f64>) {
    let n = A.nrows();
    let k = n_components.min(n);
    let mut vectors = Array2::<f64>::zeros((n, k));
    let mut values = Array1::<f64>::zeros(k);
    let mut work = A.to_owned();

    for comp in 0..k {
        let mut v = Array1::from_shape_fn(n, |_| rand::random::<f64>() - 0.5);
        let vn = v.dot(&v).sqrt();
        if vn > 0.0 { v /= vn; }
        for _ in 0..100 {
            let v_new = work.dot(&v);
            let vnn = v_new.dot(&v_new).sqrt();
            if vnn < 1e-12 { break; }
            let v_new = v_new / vnn;
            let diff = (&v_new - &v).mapv(|x| x.abs()).sum();
            v = v_new;
            if diff < 1e-8 { break; }
        }
        let eval = v.dot(&work.dot(&v));
        if eval.abs() < 1e-12 { break; }
        vectors.column_mut(comp).assign(&v);
        values[comp] = eval;
        work = &work - eval * v.dot(&v.t());
    }
    (vectors, values)
}

fn solve_linear(A: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    let n = A.nrows();
    let mut Ac = A.to_owned();
    let mut bc = b.clone();
    for col in 0..n {
        let pivot = Ac[[col, col]];
        if pivot.abs() < 1e-14 { continue; }
        for row in (col + 1)..n {
            let f = Ac[[row, col]] / pivot;
            for c in col..n { Ac[[row, c]] -= f * Ac[[col, c]]; }
            bc[row] -= f * bc[col];
        }
    }
    let mut x = Array1::<f64>::zeros(n);
    for col in (0..n).rev() {
        let pivot = Ac[[col, col]];
        if pivot.abs() > 1e-14 {
            let mut s = bc[col];
            for c in (col + 1)..n { s -= Ac[[col, c]] * x[c]; }
            x[col] = s / pivot;
        }
    }
    x
}

pub struct Isomap {
    pub n_neighbors: usize,
    pub n_components: usize,
}

impl Isomap {
    pub fn new(n_neighbors: usize, n_components: usize) -> Self {
        Self { n_neighbors, n_components }
    }

    pub fn fit_transform(&self, x: &[f64], n_samples: usize, n_features: usize) -> Result<Vec<f64>, String> {
        if x.len() != n_samples * n_features {
            return Err("Dimension mismatch".to_string());
        }
        let X = Array2::from_shape_vec((n_samples, n_features), x.to_vec()).unwrap();
        let D = pairwise_distances(&X.view());

        let mut knn = Array2::<f64>::from_elem((n_samples, n_samples), f64::INFINITY);
        for i in 0..n_samples {
            let mut neigh: Vec<(usize, f64)> = (0..n_samples).map(|j| (j, D[[i, j]])).collect();
            neigh.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            for &(j, d) in neigh.iter().take(self.n_neighbors + 1) {
                if i != j { knn[[i, j]] = d; }
            }
        }

        let mut G = knn.clone();
        for k in 0..n_samples {
            for i in 0..n_samples {
                for j in 0..n_samples {
                    if G[[i, k]] < f64::INFINITY && G[[k, j]] < f64::INFINITY {
                        let nd = G[[i, k]] + G[[k, j]];
                        if nd < G[[i, j]] { G[[i, j]] = nd; }
                    }
                }
            }
        }

        let mut J = Array2::<f64>::eye(n_samples);
        let one_n = 1.0 / n_samples as f64;
        for i in 0..n_samples { for j in 0..n_samples { J[[i, j]] -= one_n; } }
        let G_sq = G.mapv(|v| v * v);
        let mut B = J.dot(&G_sq).dot(&J);
        B *= -0.5;

        let (components, _) = power_eigen(&B, self.n_components);
        let k = self.n_components.min(n_samples);

        let mut result = Vec::with_capacity(n_samples * self.n_components);
        for i in 0..n_samples {
            for j in 0..k {
                result.push(components[[i, j]]);
            }
        }
        Ok(result)
    }
}

pub struct LocallyLinearEmbedding {
    pub n_neighbors: usize,
    pub n_components: usize,
    pub method: String,
}

impl LocallyLinearEmbedding {
    pub fn new(n_neighbors: usize, n_components: usize, method: String) -> Self {
        Self { n_neighbors, n_components, method }
    }

    pub fn fit_transform(&self, x: &[f64], n_samples: usize, n_features: usize) -> Result<Vec<f64>, String> {
        if x.len() != n_samples * n_features {
            return Err("Dimension mismatch".to_string());
        }
        let X = Array2::from_shape_vec((n_samples, n_features), x.to_vec()).unwrap();
        let D = pairwise_distances(&X.view());

        let mut W = Array2::<f64>::zeros((n_samples, n_samples));
        let k = self.n_neighbors.min(n_samples - 1);
        for i in 0..n_samples {
            let mut neigh: Vec<(usize, f64)> = (0..n_samples).map(|j| (j, D[[i, j]])).collect();
            neigh.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let mut Z = Array2::<f64>::zeros((k, k));
            for a in 0..k {
                for b in 0..k {
                    let na = neigh[a + 1].0;
                    let nb = neigh[b + 1].0;
                    Z[[a, b]] = 0.5 * (D[[i, na]] * D[[i, na]] + D[[i, nb]] * D[[i, nb]] - D[[na, nb]] * D[[na, nb]]);
                }
            }
            Z[[0, 0]] += (0..k).map(|a| Z[[a, a]]).sum::<f64>() * 1e-3;
            let ones = Array1::<f64>::from_elem(k, 1.0);
            let w = solve_linear(&Z, &ones);
            for a in 0..k {
                let na = neigh[a + 1].0;
                W[[i, na]] = w[a];
            }
            let row_sum: f64 = W.row(i).sum();
            if row_sum > 0.0 {
                for j in 0..n_samples { W[[i, j]] /= row_sum; }
            }
        }

        let mut M = Array2::<f64>::eye(n_samples);
        for i in 0..n_samples {
            let row_sum: f64 = W.row(i).sum();
            for j in 0..n_samples {
                M[[i, j]] -= W[[i, j]];
                if i == j { M[[i, j]] += row_sum; }
            }
        }

        let k = self.n_components.min(n_samples);
        let (components, _) = power_eigen(&M, k);

        let mut result = Vec::with_capacity(n_samples * self.n_components);
        for i in 0..n_samples {
            for j in 0..k {
                result.push(components[[i, j]]);
            }
        }
        Ok(result)
    }
}

pub struct TSNE {
    pub n_components: usize,
    pub perplexity: f64,
}

impl TSNE {
    pub fn new(n_components: usize, perplexity: f64) -> Self {
        Self { n_components, perplexity }
    }

    pub fn fit_transform(&self, x: &[f64], n_samples: usize, n_features: usize) -> Result<Vec<f64>, String> {
        if x.len() != n_samples * n_features {
            return Err("Dimension mismatch".to_string());
        }
        let X = Array2::from_shape_vec((n_samples, n_features), x.to_vec()).unwrap();
        let D = pairwise_distances(&X.view());

        let mut P = Array2::<f64>::zeros((n_samples, n_samples));
        let log_k = (self.perplexity * 0.5).ln();
        for i in 0..n_samples {
            let mut beta = 1.0;
            for _ in 0..50 {
                let mut sum_p = 0.0;
                for j in 0..n_samples {
                    if i != j {
                        P[[i, j]] = (-D[[i, j]] * D[[i, j]] * beta).exp();
                        sum_p += P[[i, j]];
                    }
                }
                if sum_p < 1e-12 { break; }
                let h = (sum_p.ln() - beta * D.row(i).mapv(|v| v * v).iter().filter(|&&v| v > 0.0).map(|v| *v).sum::<f64>() / sum_p).ln();
                if h.is_nan() { break; }
                beta *= h / log_k;
                if beta < 0.0 { beta = 1.0; }
            }
            let sum_p: f64 = P.row(i).sum();
            if sum_p > 0.0 { P.row_mut(i).mapv_inplace(|v| v / sum_p); }
        }
        for i in 0..n_samples {
            for j in i + 1..n_samples {
                let p = (P[[i, j]] + P[[j, i]]) / (2.0 * n_samples as f64);
                P[[i, j]] = p.max(1e-12);
                P[[j, i]] = p.max(1e-12);
            }
        }

        let mut Y = Array2::<f64>::zeros((n_samples, self.n_components));
        for i in 0..n_samples {
            for j in 0..self.n_components { Y[[i, j]] = rand::random::<f64>() * 1e-4; }
        }

        let mut Y_inc = Array2::<f64>::zeros((n_samples, self.n_components));
        let mut Y_gain = Array2::<f64>::from_elem((n_samples, self.n_components), 1.0);

        for _ in 0..1000 {
            let mut Q = Array2::<f64>::zeros((n_samples, n_samples));
            let mut sum_q = 0.0;
            for i in 0..n_samples {
                for j in i + 1..n_samples {
                    let mut d = 0.0;
                    for k in 0..self.n_components { let diff = Y[[i, k]] - Y[[j, k]]; d += diff * diff; }
                    let q = 1.0 / (1.0 + d);
                    Q[[i, j]] = q; Q[[j, i]] = q;
                    sum_q += 2.0 * q;
                }
            }
            if sum_q < 1e-12 { break; }
            Q /= sum_q;
            Q.mapv_inplace(|v| v.max(1e-12));

            let mut grad = Array2::<f64>::zeros((n_samples, self.n_components));
            for i in 0..n_samples {
                for j in 0..n_samples {
                    if i == j { continue; }
                    let diff_pq = (P[[i, j]] - Q[[i, j]]) * Q[[i, j]];
                    for k in 0..self.n_components {
                        grad[[i, k]] += 4.0 * diff_pq * (Y[[i, k]] - Y[[j, k]]);
                    }
                }
            }

            for i in 0..n_samples {
                for j in 0..self.n_components {
                    let g = grad[[i, j]];
                    Y_gain[[i, j]] = if g.signum() != Y_inc[[i, j]].signum() {
                        (Y_gain[[i, j]] + 0.2).max(0.01)
                    } else {
                        (Y_gain[[i, j]] * 0.8).max(0.01)
                    };
                    Y_inc[[i, j]] = 0.5 * Y_inc[[i, j]] - 200.0 * Y_gain[[i, j]] * g;
                    Y[[i, j]] += Y_inc[[i, j]];
                }
            }
        }

        let mut result = Vec::with_capacity(n_samples * self.n_components);
        for i in 0..n_samples {
            for j in 0..self.n_components { result.push(Y[[i, j]]); }
        }
        Ok(result)
    }
}

pub struct UMAP {
    pub n_components: usize,
    pub n_neighbors: usize,
}

impl UMAP {
    pub fn new(n_components: usize, n_neighbors: usize) -> Self {
        Self { n_components, n_neighbors }
    }

    pub fn fit_transform(&self, x: &[f64], n_samples: usize, n_features: usize) -> Result<Vec<f64>, String> {
        if x.len() != n_samples * n_features {
            return Err("Dimension mismatch".to_string());
        }
        let X = Array2::from_shape_vec((n_samples, n_features), x.to_vec()).unwrap();
        let D = pairwise_distances(&X.view());

        let mut graph = Array2::<f64>::zeros((n_samples, n_samples));
        let k = self.n_neighbors.min(n_samples - 1);
        for i in 0..n_samples {
            let mut neigh: Vec<(usize, f64)> = (0..n_samples).map(|j| (j, D[[i, j]])).collect();
            neigh.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let sigma = neigh[1..=k].iter().map(|(_, d)| d).sum::<f64>() / k as f64;
            let sigma = sigma.max(1e-12);
            for &(j, d) in neigh.iter().take(k + 1) {
                if i != j { graph[[i, j]] = (-(d - D[[i, neigh[0].0]]) / sigma).exp(); }
            }
        }
        for i in 0..n_samples {
            for j in 0..n_samples {
                graph[[i, j]] = graph[[i, j]] + graph[[j, i]] - graph[[i, j]] * graph[[j, i]];
            }
        }

        let mut embedding = Array2::<f64>::zeros((n_samples, self.n_components));
        for i in 0..n_samples {
            for j in 0..self.n_components { embedding[[i, j]] = rand::random::<f64>() * 0.01; }
        }

        for _ in 0..200 {
            let mut grad = Array2::<f64>::zeros((n_samples, self.n_components));
            for i in 0..n_samples {
                for j in 0..n_samples {
                    if graph[[i, j]] < 1e-12 { continue; }
                    let mut d = 0.0;
                    for k in 0..self.n_components { let diff = embedding[[i, k]] - embedding[[j, k]]; d += diff * diff; }
                    let q = 1.0 / (1.0 + d);
                    let w = graph[[i, j]] * q;
                    for k in 0..self.n_components {
                        let diff = embedding[[i, k]] - embedding[[j, k]];
                        grad[[i, k]] += 2.0 * w * diff;
                        grad[[j, k]] -= 2.0 * w * diff;
                    }
                }
            }
            for i in 0..n_samples {
                for k in 0..self.n_components { embedding[[i, k]] -= 1.0 * grad[[i, k]]; }
            }
        }

        let mut result = Vec::with_capacity(n_samples * self.n_components);
        for i in 0..n_samples {
            for j in 0..self.n_components { result.push(embedding[[i, j]]); }
        }
        Ok(result)
    }
}
