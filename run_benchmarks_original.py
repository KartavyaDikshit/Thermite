import time
import json
import os
import numpy as np
from sklearn.datasets import make_classification, make_regression, make_blobs
from sklearn.metrics import accuracy_score, r2_score

import sklearn.linear_model as sk_lm
import sklearn.ensemble as sk_ens
import sklearn.cluster as sk_clu
import sklearn.metrics as sk_met

import thermite.linear_model as th_lm
import thermite.ensemble as th_ens
import thermite.cluster as th_clu
import thermite.metrics as th_met

def generate_datasets(n_samples):
    X_cls, y_cls = make_classification(n_samples=n_samples, n_features=20, random_state=42)
    X_reg, y_reg = make_regression(n_samples=n_samples, n_features=20, random_state=42)
    X_blobs, y_blobs = make_blobs(n_samples=n_samples, n_features=20, centers=5, random_state=42)
    return {
        'classification': (X_cls, y_cls),
        'regression': (X_reg, y_reg),
        'clustering': (X_blobs, y_blobs),
        'distances': (X_blobs, y_blobs)
    }

configs = [
    {
        'name': 'LinearRegression',
        'task': 'regression',
        'sklearn_cls': sk_lm.LinearRegression,
        'thermite_cls': th_lm.LinearRegression,
        'sk_kwargs': {},
        'th_kwargs': {}
    },
    {
        'name': 'LogisticRegression',
        'task': 'classification',
        'sklearn_cls': sk_lm.LogisticRegression,
        'thermite_cls': th_lm.LogisticRegression,
        'sk_kwargs': {'max_iter': 100},
        'th_kwargs': {'max_iter': 100}
    },
    {
        'name': 'RandomForestClassifier',
        'task': 'classification',
        'sklearn_cls': sk_ens.RandomForestClassifier,
        'thermite_cls': th_ens.RandomForestClassifier,
        'sk_kwargs': {'n_estimators': 50, 'max_depth': 10, 'random_state': 42},
        'th_kwargs': {'n_estimators': 50, 'max_depth': 10, 'random_state': 42}
    },
    {
        'name': 'GradientBoostingRegressor',
        'task': 'regression',
        'sklearn_cls': sk_ens.GradientBoostingRegressor,
        'thermite_cls': th_ens.GradientBoostingRegressor,
        'sk_kwargs': {'n_estimators': 50, 'max_depth': 5, 'random_state': 42},
        'th_kwargs': {'n_estimators': 50, 'max_depth': 5, 'random_state': 42}
    },
    {
        'name': 'HistGradientBoostingClassifier',
        'task': 'classification',
        'sklearn_cls': sk_ens.HistGradientBoostingClassifier,
        'thermite_cls': th_ens.HistGradientBoostingClassifier,
        'sk_kwargs': {'max_iter': 50, 'random_state': 42},
        'th_kwargs': {'n_estimators': 50, 'random_state': 42}
    },
    {
        'name': 'KMeans',
        'task': 'clustering',
        'sklearn_cls': sk_clu.KMeans,
        'thermite_cls': th_clu.KMeans,
        'sk_kwargs': {'n_clusters': 5, 'random_state': 42, 'n_init': 1},
        'th_kwargs': {'n_clusters': 5, 'random_state': 42, 'n_init': 1}
    },
    {
        'name': 'MiniBatchKMeans',
        'task': 'clustering',
        'sklearn_cls': sk_clu.MiniBatchKMeans,
        'thermite_cls': th_clu.MiniBatchKMeans,
        'sk_kwargs': {'n_clusters': 5, 'random_state': 42, 'n_init': 1},
        'th_kwargs': {'n_clusters': 5}
    }
]

def run_benchmarks():
    sizes = [10000, 100000]
    all_results = []
    
    for size in sizes:
        print(f"\n--- Benchmarking size {size} ---")
        datasets = generate_datasets(size)
        
        for config in configs:
            task = config['task']
            X, y = datasets[task]
            
            # Scikit-learn
            sk_model = config['sklearn_cls'](**config['sk_kwargs'])
            t0 = time.time()
            if task != 'clustering':
                sk_model.fit(X, y)
            else:
                sk_model.fit(X)
            sk_train_time = time.time() - t0
            
            t0 = time.time()
            sk_preds = sk_model.predict(X)
            sk_infer_time = time.time() - t0
            
            # Thermite
            th_model = config['thermite_cls'](**config['th_kwargs'])
            t0 = time.time()
            if task != 'clustering':
                th_model.fit(X, y)
            else:
                th_model.fit(X)
            th_train_time = time.time() - t0
            
            t0 = time.time()
            th_preds = th_model.predict(X)
            th_infer_time = time.time() - t0
            
            # Metric
            sk_metric = 0.0
            th_metric = 0.0
            if task == 'classification':
                sk_metric = accuracy_score(y, sk_preds)
                th_metric = accuracy_score(y, th_preds)
            elif task == 'regression':
                sk_metric = r2_score(y, sk_preds)
                th_metric = r2_score(y, th_preds)
            else:
                pass # skip metric for clustering in summary
                
            res = {
                'size': size,
                'model': config['name'],
                'sk_train_time': sk_train_time,
                'th_train_time': th_train_time,
                'sk_infer_time': sk_infer_time,
                'th_infer_time': th_infer_time,
                'sk_metric': sk_metric,
                'th_metric': th_metric
            }
            print(f"{config['name']} (Size {size}):")
            print(f"  Train: SK={sk_train_time:.3f}s TH={th_train_time:.3f}s (Speedup: {sk_train_time/th_train_time if th_train_time > 0 else 0:.2f}x)")
            print(f"  Infer: SK={sk_infer_time:.3f}s TH={th_infer_time:.3f}s (Speedup: {sk_infer_time/th_infer_time if th_infer_time > 0 else 0:.2f}x)")
            all_results.append(res)
            
        # Pairwise distances benchmark
        X, _ = datasets['distances']
        X_pd = X[:min(size, 10000)] # cap to 10k to prevent OOM
        print(f"pairwise_distances (Size {len(X_pd)}):")
        
        t0 = time.time()
        sk_dist = sk_met.pairwise_distances(X_pd)
        sk_pd_time = time.time() - t0
        
        t0 = time.time()
        th_dist = th_met.pairwise_distances(X_pd, X_pd)
        th_pd_time = time.time() - t0
        
        res = {
            'size': len(X_pd),
            'model': 'pairwise_distances',
            'sk_train_time': sk_pd_time,
            'th_train_time': th_pd_time,
            'sk_infer_time': 0,
            'th_infer_time': 0,
            'sk_metric': 0,
            'th_metric': 0
        }
        all_results.append(res)
        print(f"  Time: SK={sk_pd_time:.3f}s TH={th_pd_time:.3f}s (Speedup: {sk_pd_time/th_pd_time if th_pd_time > 0 else 0:.2f}x)")

    with open('benchmarks/results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    md_content = "# Thermite vs Scikit-Learn Benchmark Results\n\n"
    md_content += "| Size | Model | SK Train (s) | TH Train (s) | Train Speedup | SK Infer (s) | TH Infer (s) | Infer Speedup | SK Metric | TH Metric |\n"
    md_content += "|---|---|---|---|---|---|---|---|---|---|\n"
    
    for r in all_results:
        train_speedup = r['sk_train_time'] / r['th_train_time'] if r['th_train_time'] > 0 else 0
        infer_speedup = r['sk_infer_time'] / r['th_infer_time'] if r['th_infer_time'] > 0 else 0
        md_content += f"| {r['size']} | {r['model']} | {r['sk_train_time']:.3f} | {r['th_train_time']:.3f} | {train_speedup:.2f}x | {r['sk_infer_time']:.3f} | {r['th_infer_time']:.3f} | {infer_speedup:.2f}x | {r['sk_metric']:.4f} | {r['th_metric']:.4f} |\n"
        
    with open('benchmarks/benchmark_results.md', 'w') as f:
        f.write(md_content)
    print("Results saved to benchmarks/benchmark_results.md")

if __name__ == '__main__':
    run_benchmarks()
