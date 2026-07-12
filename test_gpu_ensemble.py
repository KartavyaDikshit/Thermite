import numpy as np
from sklearn.datasets import make_classification, make_regression
from thermite.ensemble import RandomForestClassifier, RandomForestRegressor

X_c, y_c = make_classification(n_samples=500, n_features=20, random_state=42)
y_c = y_c.astype(float)

# CPU device
clf_cpu = RandomForestClassifier(n_estimators=50, random_state=42, device='cpu')
clf_cpu.fit(X_c, y_c)
preds_cpu = clf_cpu.predict(X_c)

# GPU device (falls back to cpu path on this machine without wgpu feature, verifies API works)
clf_gpu = RandomForestClassifier(n_estimators=50, random_state=42, device='gpu')
clf_gpu.fit(X_c, y_c)
preds_gpu = clf_gpu.predict(X_c)

print(f"CPU acc: {(preds_cpu == y_c).mean():.4f}")
print(f"GPU acc: {(preds_gpu == y_c).mean():.4f}")
print(f"Predictions identical: {np.array_equal(preds_cpu, preds_gpu)}")

# Regression
X_r, y_r = make_regression(n_samples=500, n_features=20, random_state=42)
rfr = RandomForestRegressor(n_estimators=50, random_state=42, device='cpu')
rfr.fit(X_r, y_r)
preds_r = rfr.predict(X_r)
ss_res = np.sum((y_r - preds_r)**2)
ss_tot = np.sum((y_r - y_r.mean())**2)
print(f"Regressor R2: {1 - ss_res/ss_tot:.4f}")
