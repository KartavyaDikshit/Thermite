# Thermite vs Scikit-Learn Benchmark Results

| Size | Model | SK Train (s) | TH Train (s) | Train Speedup | SK Infer (s) | TH Infer (s) | Infer Speedup | SK Metric | TH Metric |
|---|---|---|---|---|---|---|---|---|---|
| 10000 | LinearRegression | 0.002 | 0.001 | 1.96x | 0.000 | 0.000 | 0.30x | 1.0000 | 1.0000 |
| 10000 | LogisticRegression | 0.003 | 0.003 | 0.92x | 0.000 | 0.001 | 0.25x | 0.8892 | 0.8747 |
| 10000 | RandomForestClassifier | 0.560 | 0.255 | 2.20x | 0.017 | 0.004 | 4.17x | 0.9504 | 0.9495 |
| 10000 | GradientBoostingRegressor | 2.267 | 0.940 | 2.41x | 0.007 | 0.009 | 0.80x | 0.9595 | 0.9595 |
| 10000 | HistGradientBoostingClassifier | 0.279 | 0.352 | 0.79x | 0.004 | 0.006 | 0.61x | 0.9548 | 0.9231 |
| 10000 | KMeans | 0.055 | 0.005 | 10.87x | 0.000 | 0.000 | 0.81x | 0.0000 | 0.0000 |
| 10000 | MiniBatchKMeans | 0.013 | 0.035 | 0.36x | 0.000 | 0.000 | 0.84x | 0.0000 | 0.0000 |
| 10000 | pairwise_distances | 0.287 | 0.862 | 0.33x | 0.000 | 0.000 | 0.00x | 0.0000 | 0.0000 |
| 100000 | LinearRegression | 0.015 | 0.016 | 0.91x | 0.001 | 0.010 | 0.07x | 1.0000 | 1.0000 |
| 100000 | LogisticRegression | 0.016 | 0.055 | 0.30x | 0.001 | 0.008 | 0.11x | 0.8680 | 0.8613 |
| 100000 | RandomForestClassifier | 7.440 | 11.066 | 0.67x | 0.159 | 0.038 | 4.21x | 0.8800 | 0.8796 |
| 100000 | GradientBoostingRegressor | 29.750 | 19.852 | 1.50x | 0.074 | 0.100 | 0.74x | 0.9286 | 0.9286 |
| 100000 | HistGradientBoostingClassifier | 0.467 | 4.363 | 0.11x | 0.045 | 0.086 | 0.52x | 0.8770 | 0.8698 |
| 100000 | KMeans | 0.021 | 0.007 | 3.15x | 0.001 | 0.001 | 0.87x | 0.0000 | 0.0000 |
| 100000 | MiniBatchKMeans | 0.015 | 0.370 | 0.04x | 0.002 | 0.002 | 0.88x | 0.0000 | 0.0000 |
| 10000 | pairwise_distances | 0.342 | 1.040 | 0.33x | 0.000 | 0.000 | 0.00x | 0.0000 | 0.0000 |
