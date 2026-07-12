import numpy as np
from sklearn.datasets import make_classification
from thermite.naive_bayes import GaussianNB

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
classes = np.unique(y)

model1 = GaussianNB()
model1.fit(X, y)
preds1 = model1.predict(X)

model2 = GaussianNB()
# Train in batches
for i in range(0, 1000, 100):
    model2.partial_fit(X[i:i+100], y[i:i+100], classes=classes)
    
preds2 = model2.predict(X)

print("Match:", np.allclose(preds1, preds2))
