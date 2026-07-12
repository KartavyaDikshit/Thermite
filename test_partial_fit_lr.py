import numpy as np
from sklearn.datasets import make_classification
from thermite.linear_model import LogisticRegression

X, y = make_classification(n_samples=2000, n_features=20, random_state=42)
y = y.astype(float)
classes = np.array([0.0, 1.0])

# Batch fit
m1 = LogisticRegression(C=1.0, max_iter=200)
m1.fit(X, y)
acc1 = (m1.predict(X) == y).mean()

# Streaming partial_fit
m2 = LogisticRegression(C=1.0)
for i in range(0, 2000, 200):
    m2.partial_fit(X[i:i+200], y[i:i+200], classes=classes)
acc2 = (m2.predict(X) == y).mean()

print(f"Batch fit accuracy:   {acc1:.4f}")
print(f"Streaming partial_fit accuracy: {acc2:.4f}")
print("partial_fit works:" , acc2 > 0.7)
