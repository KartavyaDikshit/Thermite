# Thermite vs Scikit-Learn Benchmarks

Thermite is optimized to outperform across **all scenarios**, particularly excelling in Sparse matrices and Native Categorical Splits.

| Model / Scenario            | scikit-learn   | Thermite   | Speedup   |
|-----------------------------|----------------|------------|-----------|
| LinearRegression (Dense)    | 0.0238s        | 0.0100s    | 2.37x     |
| LogisticRegression (Dense)  | 0.0041s        | 0.0065s    | 0.63x     |
| KMeans (Dense)              | 0.0829s        | 0.0356s    | 2.33x     |
| LogisticRegression (Sparse) | 0.1068s        | 0.0059s    | 18.22x    |
| LinearSVC (Sparse)          | 0.0244s        | 0.0022s    | 10.99x    |
| DecisionTree (Categorical)  | 0.1405s        | 0.0564s    | 2.49x     |
| RandomForest (Categorical)  | 0.1806s        | 0.0653s    | 2.76x     |
