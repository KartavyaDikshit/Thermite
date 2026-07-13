# Thermite vs Scikit-Learn Benchmark Results

| Size | Model | SK Train (s) | TH Train (s) | Train Speedup | SK Infer (s) | TH Infer (s) | Infer Speedup | SK Metric | TH Metric |
|---|---|---|---|---|---|---|---|---|---|
| 10000 | LinearRegression | 0.002 | 0.001 | 1.58x | 0.000 | 0.000 | 0.78x | 1.0000 | 1.0000 |
| 10000 | LogisticRegression | 0.006 | 0.003 | 1.71x | 0.000 | 0.001 | 0.27x | 0.8892 | 0.8747 |
| 10000 | RandomForestClassifier | 0.560 | 0.249 | 2.25x | 0.016 | 0.004 | 3.80x | 0.9504 | 0.9493 |
| 10000 | GradientBoostingRegressor | 2.338 | 0.946 | 2.47x | 0.008 | 0.009 | 0.88x | 0.9595 | 0.9595 |
| 10000 | HistGradientBoostingClassifier | 0.202 | 0.357 | 0.56x | 0.003 | 0.006 | 0.57x | 0.9548 | 0.9231 |
| 10000 | KMeans | 0.015 | 0.004 | 3.59x | 0.000 | 0.000 | 1.50x | 0.0000 | 0.0000 |
| 10000 | MiniBatchKMeans | 0.015 | 0.033 | 0.45x | 0.000 | 0.000 | 2.46x | 0.0000 | 0.0000 |
| 10000 | pairwise_distances | 0.269 | 0.625 | 0.43x | 0.000 | 0.000 | 0.00x | 0.0000 | 0.0000 |
| 100000 | LinearRegression | 0.015 | 0.016 | 0.90x | 0.001 | 0.003 | 0.22x | 1.0000 | 1.0000 |
| 100000 | LogisticRegression | 0.016 | 0.056 | 0.28x | 0.001 | 0.009 | 0.09x | 0.8680 | 0.8613 |
| 100000 | RandomForestClassifier | 7.593 | 11.169 | 0.68x | 0.163 | 0.037 | 4.39x | 0.8800 | 0.8796 |
| 100000 | GradientBoostingRegressor | 28.341 | 16.550 | 1.71x | 0.071 | 0.089 | 0.79x | 0.9286 | 0.9286 |
| 100000 | HistGradientBoostingClassifier | 0.491 | 3.747 | 0.13x | 0.027 | 0.068 | 0.41x | 0.8770 | 0.8698 |
| 100000 | KMeans | 0.020 | 0.008 | 2.60x | 0.001 | 0.001 | 0.83x | 0.0000 | 0.0000 |
| 100000 | MiniBatchKMeans | 0.013 | 0.353 | 0.04x | 0.001 | 0.002 | 0.58x | 0.0000 | 0.0000 |
| 10000 | pairwise_distances | 0.297 | 0.265 | 1.12x | 0.000 | 0.000 | 0.00x | 0.0000 | 0.0000 |
