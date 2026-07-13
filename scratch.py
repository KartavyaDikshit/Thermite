import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression as sk_LR
from thermite.linear_model import LogisticRegression as th_LR
import time

X, y = make_classification(n_samples=100000, n_features=20, random_state=42)

clf1 = sk_LR(max_iter=100)
t0 = time.time()
clf1.fit(X, y)
print("SKLR time:", time.time()-t0, "iters:", clf1.n_iter_)

clf2 = th_LR(max_iter=100)
t0 = time.time()
clf2.fit(X, y)
print("THLR time:", time.time()-t0)
