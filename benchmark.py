import time
import numpy as np
from tabulate import tabulate

import sklearn.linear_model as sk_lm
import sklearn.tree as sk_tree
import sklearn.cluster as sk_cluster
import sklearn.decomposition as sk_decomp

import thermite.linear_model as th_lm
import thermite.tree as th_tree
import thermite.cluster as th_cluster
import thermite.decomposition as th_decomp

def generate_data(n_samples=10000, n_features=100, task="regression"):
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    if task == "regression":
        y = X @ np.random.randn(n_features) + np.random.randn(n_samples)
    elif task == "classification":
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
    else:
        y = None
    return X, y

def run_benchmark():
    datasets = {
        "Small": (1000, 20),
        "Medium": (10000, 100),
        "Large": (50000, 200)
    }

    models = [
        {
            "name": "LinearRegression",
            "task": "regression",
            "sk": sk_lm.LinearRegression(),
            "th": th_lm.LinearRegression()
        },
        {
            "name": "LogisticRegression",
            "task": "classification",
            "sk": sk_lm.LogisticRegression(max_iter=100),
            "th": th_lm.LogisticRegression(max_iter=100)
        },
        {
            "name": "DecisionTreeRegressor",
            "task": "regression",
            "sk": sk_tree.DecisionTreeRegressor(random_state=42, max_depth=10),
            "th": th_tree.DecisionTreeRegressor(random_state=42, max_depth=10)
        },
        {
            "name": "KMeans",
            "task": "cluster",
            "sk": sk_cluster.KMeans(n_clusters=5, n_init=1, random_state=42, max_iter=100),
            "th": th_cluster.KMeans(n_clusters=5, n_init=1, random_state=42, max_iter=100)
        },
        {
            "name": "PCA",
            "task": "unsupervised",
            "sk": sk_decomp.PCA(n_components=5, random_state=42),
            "th": th_decomp.PCA(n_components=5, random_state=42)
        }
    ]

    results = []

    for size_name, (n_samples, n_features) in datasets.items():
        print(f"\\n--- Dataset: {size_name} ({n_samples} samples, {n_features} features) ---", flush=True)
        for model in models:
            print(f"Running {model['name']}...", end=" ", flush=True)
            X, y = generate_data(n_samples, n_features, task=model["task"])
            
            # Scikit-learn
            start = time.time()
            if model["task"] == "unsupervised" or model["task"] == "cluster":
                model["sk"].fit(X)
            else:
                model["sk"].fit(X, y)
            sk_time = time.time() - start
            print(f"sk={sk_time:.4f}s", end=" ", flush=True)

            # Thermite
            start = time.time()
            if model["task"] == "unsupervised" or model["task"] == "cluster":
                model["th"].fit(X)
            else:
                model["th"].fit(X, y)
            th_time = time.time() - start
            print(f"th={th_time:.4f}s", flush=True)
            
            speedup = sk_time / th_time if th_time > 0 else float("inf")

            
            results.append([
                size_name,
                model["name"],
                f"{sk_time:.4f}s",
                f"{th_time:.4f}s",
                f"{speedup:.2f}x"
            ])

    print("\\n### Benchmark Results")
    print(tabulate(results, headers=["Dataset", "Algorithm", "Scikit-learn Time", "Thermite Time", "Speedup"], tablefmt="github"))

if __name__ == "__main__":
    run_benchmark()
