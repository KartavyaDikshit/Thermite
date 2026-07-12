from thermite._core import MLPClassifier as _MLPClassifier
import numpy as np

class MLPClassifier:
    def __init__(self, hidden_layer_sizes=(100,), learning_rate=0.001, max_iter=200, device="cpu"):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.device = device
        self._model = _MLPClassifier(
            hidden_layer_sizes=list(hidden_layer_sizes),
            learning_rate=learning_rate,
            max_iter=max_iter,
            device=device,
        )

    def fit(self, X, y):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        y = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
        if self.device == "gpu" and X.shape[0] * X.shape[1] < 10000:
            import warnings
            warnings.warn("GPU warm-up tax: dataset is too small (X.shape[0] * X.shape[1] < 10000). GPU execution may be slower than CPU.")
        self._model.fit(X, y)
        return self

    def predict(self, X):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        return self._model.predict(X)

    def predict_proba(self, X):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        return self._model.predict_proba(X)

    def score(self, X, y):
        from .metrics import accuracy_score
        return accuracy_score(y, self.predict(X))
