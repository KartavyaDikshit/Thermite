# Thermite vs Scikit-Learn Benchmark Results

| Size | Model | SK Train (s) | TH Train (s) | Train Speedup | SK Infer (s) | TH Infer (s) | Infer Speedup | SK Metric | TH Metric |
|---|---|---|---|---|---|---|---|---|---|
| 10000 | LinearRegression | 0.002 | 0.001 | 1.35x | 0.000 | 0.000 | 0.81x | 1.0000 | 1.0000 |
| 10000 | LogisticRegression | 0.003 | 0.003 | 0.80x | 0.000 | 0.000 | 0.26x | 0.8892 | 0.8747 |
| 10000 | RandomForestClassifier | 0.545 | 0.249 | 2.19x | 0.016 | 0.004 | 4.16x | 0.9504 | 0.9494 |
| 10000 | GradientBoostingRegressor | 2.262 | 0.929 | 2.43x | 0.007 | 0.009 | 0.79x | 0.9595 | 0.9595 |
| 10000 | HistGradientBoostingClassifier | 0.204 | 0.345 | 0.59x | 0.003 | 0.005 | 0.63x | 0.9548 | 0.9231 |
| 10000 | KMeans | 0.011 | 0.004 | 2.63x | 0.000 | 0.000 | 1.21x | 0.0000 | 0.0000 |
| 10000 | MiniBatchKMeans | 0.013 | 0.005 | 2.56x | 0.000 | 0.000 | 1.83x | 0.0000 | 0.0000 |
| 10000 | pairwise_distances | 0.233 | 0.327 | 0.71x | 0.000 | 0.000 | 0.00x | 0.0000 | 0.0000 |
| 100000 | LinearRegression | 0.014 | 0.013 | 1.11x | 0.001 | 0.001 | 0.78x | 1.0000 | 1.0000 |
| 100000 | LogisticRegression | 0.012 | 0.052 | 0.24x | 0.001 | 0.007 | 0.13x | 0.8680 | 0.8613 |
| 100000 | RandomForestClassifier | 7.256 | 10.459 | 0.69x | 0.158 | 0.034 | 4.72x | 0.8800 | 0.8796 |
| 100000 | GradientBoostingRegressor | 27.873 | 16.308 | 1.71x | 0.071 | 0.085 | 0.84x | 0.9286 | 0.9286 |
| 100000 | HistGradientBoostingClassifier | 0.329 | 3.692 | 0.09x | 0.027 | 0.069 | 0.39x | 0.8770 | 0.8698 |
| 100000 | KMeans | 0.017 | 0.007 | 2.27x | 0.001 | 0.001 | 1.05x | 0.0000 | 0.0000 |
| 100000 | MiniBatchKMeans | 0.013 | 0.005 | 2.50x | 0.001 | 0.002 | 0.65x | 0.0000 | 0.0000 |
| 10000 | pairwise_distances | 0.281 | 0.581 | 0.48x | 0.000 | 0.000 | 0.00x | 0.0000 | 0.0000 |
