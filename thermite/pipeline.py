class Pipeline:
    def __init__(self, steps):
        self.steps = steps

    def fit(self, X, y=None, **fit_params):
        Xt = X
        for name, transform in self.steps[:-1]:
            if hasattr(transform, "fit_transform"):
                Xt = transform.fit_transform(Xt, y, **fit_params)
            else:
                Xt = transform.fit(Xt, y, **fit_params).transform(Xt)
        self.steps[-1][1].fit(Xt, y, **fit_params)
        return self

    def predict(self, X):
        Xt = X
        for name, transform in self.steps[:-1]:
            Xt = transform.transform(Xt)
        return self.steps[-1][1].predict(Xt)

    def fit_predict(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.predict(X)

    def transform(self, X):
        Xt = X
        for name, transform in self.steps:
            Xt = transform.transform(Xt)
        return Xt
