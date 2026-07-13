import numpy as np
from thermite.linear_model import LogisticRegression as th_LR
from sklearn.linear_model import LogisticRegression as sk_LR
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=100000, n_features=20, random_state=42)

clf1 = sk_LR(max_iter=100, C=1.0)
clf1.fit(X, y)
print("SKLR C=1.0 acc:", accuracy_score(y, clf1.predict(X)))

clf2 = th_LR(max_iter=100, C=1.0)
clf2.fit(X, y)
print("THLR C=1.0 acc:", accuracy_score(y, clf2.predict(X)))

clf3 = th_LR(max_iter=100, C=100000.0)
clf3.fit(X, y)
print("THLR C=100000.0 acc:", accuracy_score(y, clf3.predict(X)))
