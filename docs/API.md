# Thermite ML - Full API Reference

Thermite ML strictly adheres to the scikit-learn API. All estimators expose `fit`, `predict`, `predict_proba`, `score`, `fit_transform`, etc.

## Enterprise Features

### 1. Model Card Generation
Generate comprehensive Markdown audits automatically.
```python
from thermite.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X, y, generate_model_card=True) # Outputs a Model Card automatically
```

### 2. Zero-Overhead Checkpointing
Bypass Python's `pickle` entirely.
```python
clf.save_checkpoint("model.bin")
```

### 3. Hardware Acceleration
Target Apple Silicon, Vulkan, or DX12 seamlessly.
```python
clf = RandomForestClassifier(device='gpu') # Routes massive reductions to wgpu Compute Shaders
```

## Module Reference

### `thermite.ensemble`
- `RandomForestClassifier` / `RandomForestRegressor`
- `GradientBoostingClassifier` / `GradientBoostingRegressor`
- `HistGradientBoostingClassifier` / `HistGradientBoostingRegressor`
- `IsolationForest`

### `thermite.linear_model`
- `LinearRegression`, `Ridge`, `Lasso`
- `LogisticRegression`
- `SGDClassifier`

### `thermite.svm`
- `SVC`: Support Vector Classification using Kernel Tricks via C-bindings.

### `thermite.causal`
- `TLearner`: CATE estimation framework.

### `thermite.federated`
- `ParameterServer`: Aggregate models iteratively.

### `thermite.rag`
- `VectorStore`: Lightweight and fast embedding search.

### `thermite.quantum`
- `QSVC`: Placeholder class wrapping Qiskit Quantum Kernels.
