# Thermite vs Scikit-Learn Benchmark Results

| Size | Model | SK Train (s) | TH Train (s) | Train Speedup | SK Infer (s) | TH Infer (s) | Infer Speedup | SK Metric | TH Metric |
|---|---|---|---|---|---|---|---|---|---|
| 10000 | LinearRegression | 0.002 | 0.001 | 1.99x | 0.000 | 0.000 | 0.88x | 1.0000 | 1.0000 |
| 10000 | LogisticRegression | 0.003 | 0.208 | 0.01x | 0.000 | 0.000 | 0.51x | 0.8892 | 0.8895 |
| 10000 | RandomForestClassifier | 0.563 | 0.202 | 2.79x | 0.016 | 0.004 | 3.98x | 0.9504 | 0.9492 |
| 10000 | GradientBoostingRegressor | 2.362 | 0.873 | 2.71x | 0.007 | 0.009 | 0.85x | 0.9595 | 0.9595 |
| 10000 | HistGradientBoostingClassifier | 0.233 | 0.073 | 3.19x | 0.004 | 0.006 | 0.61x | 0.9548 | 0.9231 |
| 10000 | KMeans | 0.012 | 0.005 | 2.69x | 0.000 | 0.000 | 1.50x | 0.0000 | 0.0000 |
| 10000 | MiniBatchKMeans | 0.011 | 0.004 | 2.62x | 0.000 | 0.000 | 1.86x | 0.0000 | 0.0000 |
| 10000 | pairwise_distances | 0.250 | 0.216 | 1.16x | 0.000 | 0.000 | 0.00x | 0.0000 | 0.0000 |
| 100000 | LinearRegression | 0.015 | 0.003 | 4.65x | 0.001 | 0.001 | 0.73x | 1.0000 | 1.0000 |
| 100000 | LogisticRegression | 0.012 | 0.008 | 1.63x | 0.001 | 0.001 | 0.82x | 0.8680 | 0.8680 |
| 100000 | RandomForestClassifier | 7.532 | 2.368 | 3.18x | 0.168 | 0.036 | 4.62x | 0.8800 | 0.8796 |
| 100000 | GradientBoostingRegressor | 28.437 | 11.798 | 2.41x | 0.070 | 0.087 | 0.80x | 0.9286 | 0.9286 |
| 100000 | HistGradientBoostingClassifier | 0.421 | 0.750 | 0.56x | 0.031 | 0.067 | 0.47x | 0.8770 | 0.8698 |
| 100000 | KMeans | 0.017 | 0.007 | 2.41x | 0.001 | 0.001 | 0.85x | 0.0000 | 0.0000 |
| 100000 | MiniBatchKMeans | 0.013 | 0.005 | 2.64x | 0.002 | 0.002 | 1.03x | 0.0000 | 0.0000 |
| 10000 | pairwise_distances | 0.298 | 0.172 | 1.73x | 0.000 | 0.000 | 0.00x | 0.0000 | 0.0000 |
