# Thermite Implementation Status

> **Last updated**: 2026-07-15
> **Version**: 0.1.0

This document tracks the implementation status of every module in Thermite. It is the source of truth for what works, what is partial, and what is a stub.

---

## ✅ Real Implementations

These modules have genuine mathematical implementations and produce correct results.

| Module | File | Lines | Algorithm |
|--------|------|-------|-----------|
| Linear Models | `linear_model.rs` | 2,285 | OLS via QR, Ridge via Cholesky, Lasso via coordinate descent + ISTA, Logistic via Newton-CG + IRLS, ElasticNet, SGDClassifier |
| Decision Trees | `tree.rs` | 1,367 | Gini/Entropy/MSE split criteria, native categorical, NaN routing, flat node array |
| Random Forest / GB | `ensemble.rs` | 746 | Parallel training via Rayon, bootstrap sampling, sequential boosting |
| KMeans / DBSCAN / Spectral | `cluster.rs` | 1,037 | k-means++, batch iteration, N^2 distance, affinity + eigen decomposition |
| PCA / TruncatedSVD | `decomposition.rs` | 420 | SVD-based, full/randomized, explained variance ratio |
| KNN / LOF | `neighbors.rs` | 422 | Brute-force + kd-tree, radius neighbors |
| SVM | `svm.rs` + `svm.cpp` | 734 | C++ libsvm FFI with RBF/Poly kernels, Platt scaling |
| Preprocessing | `preprocessing.rs` | 774 | StandardScaler, MinMaxScaler, OneHotEncoder, etc. |
| Metrics | `metrics.rs` | 676 | Full sklearn-compatible metrics suite |
| Neural Network | `neural_network.rs` | 213 | MLP with backpropagation |
| Naive Bayes | `naive_bayes.rs` | 235 | GaussianNB with log-probability scoring |
| Text | `text.rs` | 160 | CountVectorizer, TfidfVectorizer, Word2Vec |
| GPU Backend | `thermite-gpu/src/lib.rs` | 628 | wgpu compute shaders for matmul, activations, ensemble voting |
| C Compiler | `compiler.rs` | 121 | Transpiles trees/forests/boosting to standalone C |
| PyO3 Bindings | `thermite-binding/src/lib.rs` | 478 | Full sklearn-compatible Python API |

---

## ⚠️ Partial / Simplified Implementations

These modules have working structure but simplified or incomplete algorithms.

| Module | File | Lines | What's Missing |
|--------|------|-------|----------------|
| Model Selection | `model_selection.rs` | 413 | GridSearchCV is a simple loop, no parallelization |
| Hyperband | `hyperband.rs` | 89 | SuccessiveHalvingSearch — basic structure |
| Feature Selection | `feature_selection.rs` | 130 | RFE uses dummy importance, SequentialFeatureSelector picks first N |
| Time Series | `time_series.rs` | 47 | AR sets all lag coefficients to 1/lags |
| AutoML | `automl.rs` | 35 | SurrogateOptimizer wraps Ridge, suggest_next does argmax |
| Multi-output | `multi_output.rs` | 47 | Simple multi-target wrapper |
| Causal | `causal.rs` | 59 | TLearner wraps two LinearRegression models |
| Federated | `federated.rs` | 52 | FederatedAveraging aggregates SGD weights |
| RAG | `rag.rs` | 42 | VectorStore with cosine distance search |

---

## ❌ ~~Stubs (Placeholders)~~ **All Fixed**

All stub modules have been replaced with real implementations as of 2026-07-15 (Dr. FixIt overhaul).

| Module | File | Lines | Algorithm |
|--------|------|-------|-----------|
| Manifold | `manifold.rs` | ~430 | Isomap (kNN graph + Floyd-Warshall + MDS), LLE (local covariance + eigen decomposition), t-SNE (gradient descent on KL divergence), UMAP (fuzzy simplicial sets + cross-entropy) |
| Mixture | `mixture.rs` | ~240 | GaussianMixture EM algorithm (E-step: multivariate Gaussian log-prob, M-step: means/covariances/weights, log-likelihood convergence) |
| Survival | `survival.rs` | ~230 | SurvivalForest with log-rank split criterion, Nelson-Aalen cumulative hazard estimator at leaves, bootstrap ensemble averaging |
| Cross Decomposition | `cross_decomposition.rs` | ~290 | PLSRegression via NIPALS iterative algorithm, CCA via power-iteration SVD of cross-covariance matrix |
| Graph | `graph.rs` | ~130 | Node2Vec with 2nd-order random walks, Skip-gram training with negative sampling, SGD optimization |
| Recommender | `recommender.rs` | ~160 | ALS with alternating least squares: closed-form solve for user/item factors via Gaussian elimination |
| Quantum | `quantum.py` | removed | Removed — was returning `np.zeros` |

---

## Test Status

- **Rust unit tests**: Only `linear_model.rs`, `tree.rs`, `svm.rs` have `#[cfg(test)]` blocks
- **Python tests**: 29 test files across 3 tiers, ~40+ tests currently skipped (edge cases)
- **CI**: GitHub Actions configured for Rust lint + Python build/test on ubuntu-latest
- **Release**: Multi-platform wheel building (macOS, Windows, Linux) via maturin