from thermite._core import IterativeImputer as _IterativeImputer

class IterativeImputer:
    def __init__(self, max_iter: int = 10):
        self._core = _IterativeImputer(max_iter=max_iter)
        self.max_iter = max_iter

    def fit_transform(self, X):
        return self._core.fit_transform(X)
