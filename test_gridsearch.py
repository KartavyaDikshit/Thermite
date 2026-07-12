import numpy as np
import time
from sklearn.datasets import make_classification
from thermite import LogisticRegression
from thermite.model_selection import GridSearchCV
import thermite

X, y = make_classification(n_samples=5000, n_features=50, random_state=42)

# Warmup / single thread
print("Running GridSearchCV with n_jobs=1")
start = time.time()
gs_single = GridSearchCV(LogisticRegression(penalty="l2"), {"C": [0.01, 0.1, 1.0, 10.0]}, cv=5, n_jobs=1)
gs_single.fit(X, y)
print(f"Time (n_jobs=1): {time.time() - start:.3f}s")

# Multi thread
print("Running GridSearchCV with n_jobs=4")
start = time.time()
gs_multi = GridSearchCV(LogisticRegression(penalty="l2"), {"C": [0.01, 0.1, 1.0, 10.0]}, cv=5, n_jobs=4)
gs_multi.fit(X, y)
print(f"Time (n_jobs=4): {time.time() - start:.3f}s")
