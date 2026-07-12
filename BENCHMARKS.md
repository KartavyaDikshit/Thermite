# Thermite vs Scikit-Learn Benchmarks

Thermite is optimized to outperform across **all scenarios**, particularly excelling in Sparse matrices and Native Categorical Splits.

| Model / Scenario            | scikit-learn   | Thermite   | Speedup   |
|-----------------------------|----------------|------------|-----------|
| LinearRegression (Dense)    | 0.0269s        | 0.0104s    | 2.58x     |
| LogisticRegression (Dense)  | 0.0075s        | 0.0067s    | 1.12x     |
| KMeans (Dense)              | 0.0859s        | 0.0366s    | 2.35x     |
| LogisticRegression (Sparse) | 0.1119s        | 0.0061s    | 18.43x    |
| LinearSVC (Sparse)          | 0.0232s        | 0.0022s    | 10.78x    |
| DecisionTree (Categorical)  | 0.1384s        | 0.0568s    | 2.44x     |
| RandomForest (Categorical)  | 0.1812s        | 0.0652s    | 2.78x     |
| GaussianNB (Dense)          | 0.0066s        | 0.0039s    | 1.69x     |
