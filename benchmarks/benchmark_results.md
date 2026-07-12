# Thermite vs Scikit-Learn Benchmark Results

| Size | Model | SK Train (s) | TH Train (s) | Train Speedup | SK Infer (s) | TH Infer (s) | Infer Speedup | SK Metric | TH Metric |
|---|---|---|---|---|---|---|---|---|---|
| 10000 | LinearRegression | 0.002 | 0.066 | 0.03x | 0.000 | 0.017 | 0.01x | 1.0000 | 1.0000 |
| 10000 | LogisticRegression | 0.003 | 0.144 | 0.02x | 0.000 | 0.015 | 0.01x | 0.8892 | 0.8747 |
| 10000 | RandomForestClassifier | 0.545 | 3.985 | 0.14x | 0.016 | 0.046 | 0.35x | 0.9504 | 0.9497 |
| 10000 | GradientBoostingRegressor | 2.229 | 43.783 | 0.05x | 0.007 | 0.103 | 0.07x | 0.9595 | 0.9595 |
| 10000 | HistGradientBoostingClassifier | 0.206 | 20.007 | 0.01x | 0.005 | 0.086 | 0.06x | 0.9548 | 0.9231 |
| 10000 | KMeans | 0.016 | 0.178 | 0.09x | 0.000 | 0.004 | 0.10x | 0.0000 | 0.0000 |
| 10000 | MiniBatchKMeans | 0.012 | 6.802 | 0.00x | 0.000 | 0.051 | 0.01x | 0.0000 | 0.0000 |
| 10000 | pairwise_distances | 0.256 | 15.050 | 0.02x | 0.000 | 0.000 | 0.00x | 0.0000 | 0.0000 |
| 100000 | LinearRegression | 0.014 | 0.683 | 0.02x | 0.001 | 0.176 | 0.00x | 1.0000 | 1.0000 |
| 100000 | LogisticRegression | 0.012 | 1.459 | 0.01x | 0.001 | 0.162 | 0.00x | 0.8680 | 0.8613 |
| 100000 | RandomForestClassifier | 7.219 | 63.822 | 0.11x | 0.158 | 0.488 | 0.32x | 0.8800 | 0.8797 |
| 100000 | GradientBoostingRegressor | 28.115 | 580.466 | 0.05x | 0.070 | 1.024 | 0.07x | 0.9286 | 0.9286 |
| 100000 | HistGradientBoostingClassifier | 0.333 | 196.233 | 0.00x | 0.027 | 0.892 | 0.03x | 0.8770 | 0.8698 |
| 100000 | KMeans | 0.019 | 0.434 | 0.04x | 0.001 | 0.035 | 0.03x | 0.0000 | 0.0000 |
| 100000 | MiniBatchKMeans | 0.013 | 68.144 | 0.00x | 0.001 | 0.511 | 0.00x | 0.0000 | 0.0000 |
| 10000 | pairwise_distances | 0.281 | 15.023 | 0.02x | 0.000 | 0.000 | 0.00x | 0.0000 | 0.0000 |
